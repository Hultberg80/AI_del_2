import streamlit as st
import pickle
import faiss
import numpy as np
import os
import cv2
import base64
import tempfile
from openai import OpenAI
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o"

# Load FAISS index and chunks with metadata
@st.cache_resource
def load_data():
    index = faiss.read_index("handboll_faiss.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    
    # Try to load metadata (source info for each chunk)
    try:
        with open("chunks_with_metadata.pkl", "rb") as f:
            data = pickle.load(f)
            metadata = data.get("metadata", None)
    except:
        metadata = None
    
    return index, chunks, metadata

index, chunks, chunk_metadata = load_data()

# Function to embed a query
def embed_query(q: str):
    e = client.embeddings.create(model=EMBED_MODEL, input=q).data[0].embedding
    e = np.array([e], dtype="float32")  # Wrap in outer array for FAISS
    faiss.normalize_L2(e)
    return e

def retrieve(query: str, top_k: int):
    query_emb = embed_query(query)
    scores, ids = index.search(query_emb, top_k)
    hits = []
    for i, s in zip(ids[0], scores[0]):
        idx = int(i)
        # Get source info if metadata exists
        if chunk_metadata and idx < len(chunk_metadata):
            source = chunk_metadata[idx].get("source", "Okänd")
            language = chunk_metadata[idx].get("language", "?")
        else:
            source = "Okänd"
            language = "?"
        hits.append((idx, float(s), chunks[idx], source, language))
    return hits

def rag_answer(query: str, top_k: int, temperature: float, history, strict_mode: bool, answer_style: str):
    hits = retrieve(query, top_k=top_k)
    context = "\n\n---\n\n".join([chunk for _, _, chunk, _, _ in hits])

    # Set up system prompt based on answer style
    if answer_style == "Kort":
        style_text = "Svara mycket kort: 1–2 meningar."
    elif answer_style == "Förklarande":
        style_text = "Svara steg-för-steg och lite mer förklarande (max 8 meningar)."
    else:
        style_text = "Svara kort först, och lägg sen en kort förklaring (3–6 meningar)."

    system = (
        "Du är en hjälpsam kompis som kan handbollsregler. "
        "Svara på svenska med enkel, avslappnad ton. "
        f"{style_text} "
        "Använd bara information från KONTEXT. "
    )

    if strict_mode:
        system += "Om svaret inte finns i KONTEXT: säg 'Jag hittar inget om det i dokumentet.'"
    else:
        system += "Om du är osäker: säg att du inte hittar det tydligt i dokumentet."

    # Incorporate recent chat history
    recent_history = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]

    messages = [{"role": "system", "content": system}]
    messages += recent_history
    messages.append({"role": "user", "content": f"FRÅGA:\n{query}\n\nKONTEXT:\n{context}\n\nSVAR:"})

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        messages=messages,
    )
    return resp.choices[0].message.content, hits


# ========== VIDEO DOMARE FUNKTIONER ==========

def download_youtube_video(url: str) -> str:
    """Ladda ner YouTube-video och returnera sökväg till filen"""
    from yt_dlp import YoutubeDL
    
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "handball_video.mp4")
    
    ydl_opts = {
        'format': 'best[height<=480][ext=mp4]/best[height<=480]/worst',  # Prova flera format
        'outtmpl': output_path,
        'overwrites': True,
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'force_generic_extractor': False,
    }
    
    # Ta bort gammal fil om den finns
    if os.path.exists(output_path):
        os.remove(output_path)
    
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # Verifiera att filen laddades ner
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise Exception("Kunde inte ladda ner videon. Prova en annan video eller ladda upp en lokal fil.")
    
    return output_path


def extract_frames_from_uploaded(uploaded_file, num_frames: int = 5) -> list:
    """Extrahera frames från en uppladdad video"""
    # Spara uppladdad fil temporärt
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "uploaded_video.mp4")
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return extract_frames(temp_path, num_frames)


def extract_frames(video_path: str, num_frames: int = 5) -> list:
    """Extrahera frames jämnt fördelade över videon"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return []
    
    # Välj frames jämnt fördelade
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Konvertera BGR till RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames


def encode_frame_to_base64(frame) -> str:
    """Konvertera frame till base64-sträng"""
    # Konvertera tillbaka till BGR för encoding
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def analyze_video_situation(frames: list) -> str:
    """Låt GPT-4o analysera frames och beskriva situationen"""
    
    # Skapa meddelande med alla frames
    content = [
        {"type": "text", "text": 
         "Du tittar på frames från ett handbollsklipp. "
         "Beskriv detaljerat vad som händer i situationen. "
         "Fokusera på: antal steg spelaren tar (max tre steg), kontakt mellan spelare, "
         "om spelaren trampar i målgården (övertramp), "
         "felaktigt byte av spelare, eller andra regelbrott. "
         "Om en försvarare står stilla och anfallaren (med boll), springer/hoppar in i hen,"
         " så är det offensiv foul(stürmerfoul), frikast för försvaret. "
         "Offensiv foul bedöms när försvararen står på ytan först och anfallaren sedan springer in i hen. "
         "Gör inte din bedömning baserat på vad domaren gör i klippet, utan bara på vad du ser i frames. "
         "Var specifik och objektiv."}
    ]
    
    for i, frame in enumerate(frames):
        b64 = encode_frame_to_base64(frame)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
        })
    
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=500
    )
    
    return response.choices[0].message.content


def get_referee_verdict(situation: str, top_k: int = 8) -> tuple:
    """Hämta relevanta regler och ge ett domslut"""
    
    # Hämta relevanta regler baserat på situationen
    hits = retrieve(situation, top_k=top_k)
    regler_kontext = "\n\n---\n\n".join([chunk for _, _, chunk, _, _ in hits])
    
    system_prompt = """Du är en erfaren handbollsdomare med djup kunskap om regelverket.

Din uppgift är att:
1. Analysera situationen som beskrivs
2. Identifiera eventuella regelbrott
3. Ge ett tydligt domslut med motivering baserat på reglerna

Var tydlig och pedagogisk i din förklaring. Använd ENDAST reglerna som ges i kontexten."""

    user_prompt = f"""SITUATIONSBESKRIVNING:
{situation}

RELEVANTA REGLER:
{regler_kontext}

DOMSLUT:
1. Vad har hänt?
2. Vilket/vilka regelbrott (om några)?
3. Mitt domslut och bestraffning
4. Motivering baserat på reglerna"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return response.choices[0].message.content, hits


# ========== STREAMLIT UI ==========

st.title("Handbollsregelassistent")

# Tabs för olika funktioner
tab1, tab2 = st.tabs(["Regelchatt", "Videodomaren"])

# Sidebar controls (globala)
st.sidebar.header("Inställningar")
answer_style = st.sidebar.selectbox("Svarslängd", ["Normal", "Kort", "Förklarande"], index=0)
strict_mode = st.sidebar.checkbox("Strikt läge (hitta inte på)", value=True)

top_k = st.sidebar.slider("Top-K (antal chunks)", 1, 15, 6) # default 6

# Temperature slider - disabled in strict mode
if strict_mode:
    st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.05, disabled=True, help="Låst till 0 i strikt läge")
    temperature = 0.0
else:
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.4, 0.05)

show_sources = st.sidebar.checkbox("Visa källor", value=False)

if st.sidebar.button("Rensa chatten"):
    st.session_state.history = []
    st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []


# ========== TAB 1: REGELCHATT ==========
with tab1:
    # Display chat history FIRST
    if st.session_state.history:
        for m in st.session_state.history:
            if m["role"] == "user":
                # User message aligned to the right
                col1, col2 = st.columns([3, 2])
                with col2:
                    st.markdown(f"**Du:** {m['content']}")
            else:
                # Assistant message aligned to the left
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Regelassistent:** {m['content']}")
                
                # Show sources for the last assistant message if checkbox is checked
                if show_sources and m == st.session_state.history[-1]:
                    # Re-retrieve to get hits for the last user query
                    last_query = st.session_state.history[-2]["content"]
                    hits = m.get("hits", [])
                    with col1:
                        with st.expander("Visa källor"):
                            for hit in hits:
                                # Handle both old format (3 values) and new format (5 values)
                                if len(hit) == 5:
                                    idx, score, chunk, source, lang = hit
                                    lang_emoji = "🇸🇪" if lang == "sv" else "🇬🇧"
                                    st.write(f"**{source}** {lang_emoji} | relevans: {score:.3f}")
                                else:
                                    idx, score, chunk = hit[:3]
                                    st.write(f"**chunk {idx}** | relevans: {score:.3f}")
                                st.caption(chunk[:600] + ("..." if len(chunk) > 600 else ""))
        
        st.divider()

    # User input at the bottom - using form for Enter key support
    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input("Skriv din fråga om handboll här:")
        submitted = st.form_submit_button("Fråga")
        
        if submitted and query:
            answer, hits = rag_answer(
                query, 
                top_k=top_k, 
                temperature=temperature, 
                history=st.session_state.history, 
                strict_mode=strict_mode, 
                answer_style=answer_style
            )

            st.session_state.history.append({"role": "user", "content": query})
            st.session_state.history.append({"role": "assistant", "content": answer, "hits": hits})
            
            # Force rerun to show the new message
            st.rerun()


# ========== TAB 2: VIDEODOMAREN ==========
with tab2:
    st.subheader("Videodomaren")
    st.write("Analysera en handbollssituation och få ett domslut baserat på officiella regler!")
    
    # Session state för video-analys
    if "video_analyzed" not in st.session_state:
        st.session_state.video_analyzed = False
    if "video_situation" not in st.session_state:
        st.session_state.video_situation = ""
    if "video_verdict" not in st.session_state:
        st.session_state.video_verdict = ""
    if "video_frames" not in st.session_state:
        st.session_state.video_frames = []
    
    # Två alternativ: YouTube eller lokal fil
    input_method = st.radio("Välj videokälla:", ["YouTube-länk", "Ladda upp video"], horizontal=True)
    
    youtube_url = ""
    uploaded_file = None
    
    if input_method == "YouTube-länk":
        youtube_url = st.text_input("YouTube-URL:", placeholder="https://www.youtube.com/watch?v=...")
        st.caption("OBS: YouTube kan ibland blockera nedladdning. Ladda upp en lokal fil om det inte fungerar.")
    else:
        uploaded_file = st.file_uploader("Välj en videofil:", type=["mp4", "mov", "avi", "mkv"])
    
    col1, col2 = st.columns(2)
    with col1:
        num_frames = st.slider("Antal frames att analysera", 3, 10, 5)
    with col2:
        video_top_k = st.slider("Antal regler att hämta", 3, 15, 8)
    
    if st.button("Analysera video", type="primary"):
        frames = []
        
        try:
            if input_method == "YouTube-länk" and youtube_url:
                with st.spinner("Laddar ner video från YouTube..."):
                    video_path = download_youtube_video(youtube_url)
                    st.success("Video nedladdad!")
                
                with st.spinner("Extraherar frames..."):
                    frames = extract_frames(video_path, num_frames=num_frames)
                    
            elif input_method == "Ladda upp video" and uploaded_file:
                with st.spinner("Bearbetar uppladdad video..."):
                    frames = extract_frames_from_uploaded(uploaded_file, num_frames=num_frames)
            else:
                st.warning("Ange en YouTube-URL eller ladda upp en videofil först!")
                st.stop()
            
            if not frames:
                st.error("Kunde inte extrahera frames från videon.")
                st.stop()
                
            st.session_state.video_frames = frames
            st.success(f"Extraherade {len(frames)} frames!")
            
            with st.spinner("Analyserar situationen med GPT-4o Vision..."):
                situation = analyze_video_situation(frames)
                st.session_state.video_situation = situation
            
            with st.spinner("Hämtar regler från ditt regeldokument och ger domslut..."):
                verdict, hits = get_referee_verdict(situation, top_k=video_top_k)
                st.session_state.video_verdict = verdict
                st.session_state.video_hits = hits
            
            st.session_state.video_analyzed = True
            st.rerun()
            
        except Exception as e:
            st.error(f"Fel vid analys: {str(e)}")
    
    # Visa resultat om analys är klar
    if st.session_state.video_analyzed:
        st.divider()
        
        # Visa frames
        st.subheader("Analyserade frames")
        cols = st.columns(min(len(st.session_state.video_frames), 5))
        for i, frame in enumerate(st.session_state.video_frames[:5]):
            with cols[i]:
                st.image(frame, caption=f"Frame {i+1}", use_container_width=True)
        
        # Visa situationsbeskrivning
        st.subheader("Situationsbeskrivning (från GPT-4o Vision)")
        st.info(st.session_state.video_situation)
        
        # Visa domslut
        st.subheader("Domslut")
        st.success(st.session_state.video_verdict)
        
        # ALLTID visa vilka regler som användes
        st.subheader("Använda regler från ditt regeldokument")
        if "video_hits" in st.session_state:
            for hit in st.session_state.video_hits:
                if len(hit) == 5:
                    idx, score, chunk, source, lang = hit
                    lang_emoji = "🇸🇪" if lang == "sv" else "🇬🇧"
                    with st.expander(f"**{source}** {lang_emoji} | Relevans: {score:.3f}"):
                        st.write(chunk)
                else:
                    idx, score, chunk = hit[:3]
                    with st.expander(f"Regel-chunk {idx} | Relevans: {score:.3f}"):
                        st.write(chunk)
        
        # Knapp för att rensa
        if st.button("Rensa analys och börja om"):
            st.session_state.video_analyzed = False
            st.session_state.video_situation = ""
            st.session_state.video_verdict = ""
            st.session_state.video_frames = []
            if "video_hits" in st.session_state:
                del st.session_state.video_hits
            st.rerun()

