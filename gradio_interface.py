"""
Gradio interface for HiggsAudio model serving.
Provides a web UI for interacting with the HiggsAudio model.
"""

import gradio as gr
import torch
import logging
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine

logger = logging.getLogger(__name__)

# Global variables to store the engine and initialization state
engine = None
is_initialized = False

def initialize_engine(
    model_path: str,
    audio_tokenizer_path: str,
    tokenizer_path: str = None,
    device: str = "cuda",
    load_in_8bit: bool = False
):
    """
    Initialize the HiggsAudio serving engine.
    
    Args:
        model_path: Path to the HiggsAudio model
        audio_tokenizer_path: Path to the audio tokenizer
        tokenizer_path: Path to the tokenizer (optional)
        device: Device to use for inference
        load_in_8bit: Whether to load model in 8-bit quantized mode
    
    Returns:
        Status message indicating success or failure
    """
    global engine, is_initialized
    
    try:
        logger.info(f"Initializing HiggsAudio engine with 8-bit: {load_in_8bit}")
        
        engine = HiggsAudioServeEngine(
            model_name_or_path=model_path,
            audio_tokenizer_name_or_path=audio_tokenizer_path,
            tokenizer_name_or_path=tokenizer_path,
            device=device,
            load_in_8bit=load_in_8bit
        )
        
        is_initialized = True
        memory_info = ""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            memory_info = f" (GPU Memory: {memory_used:.2f}GB)"
        
        quantization_info = " with 8-bit quantization" if load_in_8bit else ""
        return f"✅ Engine initialized successfully{quantization_info}{memory_info}"
        
    except Exception as e:
        is_initialized = False
        engine = None
        return f"❌ Failed to initialize engine: {str(e)}"

def create_gradio_interface():
    """Create the Gradio interface for HiggsAudio."""
    
    with gr.Blocks(title="HiggsAudio Interface") as interface:
        gr.Markdown("# HiggsAudio Model Interface")
        gr.Markdown("Configure and initialize the HiggsAudio model for audio generation.")
        
        with gr.Tab("Model Configuration"):
            with gr.Row():
                with gr.Column():
                    model_path = gr.Textbox(
                        label="Model Path",
                        placeholder="Enter model name or path (e.g., bosonai/higgs-audio-v2)",
                        value="bosonai/higgs-audio-v2"
                    )
                    
                    audio_tokenizer_path = gr.Textbox(
                        label="Audio Tokenizer Path", 
                        placeholder="Enter audio tokenizer name or path",
                        value="bosonai/higgs-audio-tokenizer"
                    )
                    
                    tokenizer_path = gr.Textbox(
                        label="Tokenizer Path (Optional)",
                        placeholder="Leave empty to use model path"
                    )
                
                with gr.Column():
                    device = gr.Dropdown(
                        choices=["cuda", "cpu"],
                        value="cuda" if torch.cuda.is_available() else "cpu",
                        label="Device"
                    )
                    
                    # Memory Management section
                    gr.Markdown("### Memory Management")
                    load_in_8bit = gr.Checkbox(
                        label="Enable 8-bit Quantization",
                        value=False,
                        info="Reduces memory usage by ~50% but may slightly impact quality. Requires bitsandbytes library."
                    )
                    
                    gr.Markdown(
                        """
                        **8-bit Quantization Benefits:**
                        - Reduces GPU memory usage by approximately 50%
                        - Enables running larger models on smaller GPUs
                        - Minimal impact on audio quality
                        - Automatically disables CUDA graph capture for compatibility
                        """
                    )
            
            with gr.Row():
                initialize_btn = gr.Button("Initialize Engine", variant="primary")
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Engine not initialized"
                )
            
            # Connect the initialization button to the function
            initialize_btn.click(
                fn=initialize_engine,
                inputs=[model_path, audio_tokenizer_path, tokenizer_path, device, load_in_8bit],
                outputs=[status_text]
            )
        
        with gr.Tab("Audio Generation"):
            gr.Markdown("Audio generation interface will be available after engine initialization.")
            # Future audio generation controls would go here
    
    return interface

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )