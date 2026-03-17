import whisper
import gradio as gr

model = whisper.load_model("small")

def transcribe(audio):

    if audio is None:
        return "Please upload audio"

    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # ✅ Force English
    result = model.transcribe(audio, language="en")

    return result["text"]
    
    
 
gr.Interface(
    title = 'OpenAI Whisper ASR Gradio Web UI', 
    fn=transcribe, 
    inputs= gr.Audio(type="filepath", label="Record or Upload Audio"),
    outputs= gr.Textbox(),
    live=True).launch()
