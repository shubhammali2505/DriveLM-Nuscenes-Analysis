
import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from evaluation.evaluator import RAGEvaluator

from evaluation.visualizer import EvaluationVisualizer
import logging
import time
import subprocess
import sys
import os
import threading
from typing import Dict, List, Any, Optional
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DriveLM Analysis Suite",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


try:
    from RAG import setup_and_run_rag_system
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    st.warning("RAG system not available. Please ensure RAG package is installed.")


class DriveLMIntegratedApp:
    
    def __init__(self):
        """Initialize the application."""
        self.output_dir = Path("results")
        self.rag_pipeline = None
        
        if not RAG_AVAILABLE:
            st.warning("⚠️ RAG system not available. Some features will be limited.")
        
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        if 'pipeline_executed' not in st.session_state:
            st.session_state.pipeline_executed = False
        if 'pipeline_running' not in st.session_state:
            st.session_state.pipeline_running = False
        if 'rag_loaded' not in st.session_state:
            st.session_state.rag_loaded = False
        if 'finetuning_completed' not in st.session_state:
            st.session_state.finetuning_completed = False
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'pipeline_output' not in st.session_state:
            st.session_state.pipeline_output = []
    
    def check_pipeline_files(self) -> Dict[str, bool]:
        files_status = {}
        
        files_to_check = [
            "unified.json",
            "analysis_results.json", 
            "findings.md",
            "analysis_summary.txt",
            "unified_rag_enhanced.json",
            "unified_rag_enhanced_statistics.json"
        ]
        
        for filename in files_to_check:
            files_status[filename] = (self.output_dir / filename).exists()
        
        return files_status
    
    def run_main_pipeline(self) -> bool:
        try:
            st.session_state.pipeline_running = True
            st.session_state.pipeline_output = []
            
            cmd = [
            sys.executable, "main.py",
            "--nuscenes_path", "../data/nusccens",  
            "--drivelm_path", "../data/drivelm_data/train_sample.json",  
            "--output_dir", "results" 
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            output_lines = []
            for line in process.stdout:
                output_lines.append(line.strip())
                st.session_state.pipeline_output = output_lines[-50:] 
                
                if len(output_lines) % 10 == 0:  
                    time.sleep(0.1)
            
            return_code = process.wait()
            
            if return_code == 0:
                st.session_state.pipeline_executed = True
                logger.info("Pipeline executed successfully")
                return True
            else:
                logger.error(f"Pipeline failed with return code: {return_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error running pipeline: {e}")
            return False
        finally:
            st.session_state.pipeline_running = False
    
    def render_sidebar(self):
        st.sidebar.title("🚗 DriveLM Analysis")
        st.sidebar.markdown("---")
        
        st.sidebar.subheader("🔄 Pipeline Status")
        
        if st.session_state.pipeline_running:
            st.sidebar.warning("⏳ Pipeline Running...")
        elif st.session_state.pipeline_executed:
            st.sidebar.success("✅ Pipeline Completed")
        else:
            st.sidebar.info("🔴 Pipeline Not Run")
        
        st.sidebar.subheader("📊 Generated Files")
        files_status = self.check_pipeline_files()
        
        for filename, exists in files_status.items():
            if exists:
                st.sidebar.success(f"✅ {filename}")
            else:
                st.sidebar.error(f"❌ {filename}")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 RAG System")
        
        if st.session_state.rag_loaded:
            st.sidebar.success("✅ RAG System Loaded")
        else:
            st.sidebar.warning("⚠️ RAG System Not Loaded run pipeline to load")
        
        if st.session_state.finetuning_completed:
            st.sidebar.success("✅ Fine-tuning Completed")
        else:
            st.sidebar.info("ℹ️ Fine-tuning Available")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Quick Actions")
        
        if st.sidebar.button("🔄 Refresh Status"):
            st.rerun()
        
        if st.sidebar.button("📁 Check Output Folder"):
            st.sidebar.info(f"Output directory: {self.output_dir.absolute()}")
            if self.output_dir.exists():
                files = list(self.output_dir.glob("*"))
                st.sidebar.write(f"Files found: {len(files)}")
    
    def render_pipeline_tab(self):
        st.header("🔄 Pipeline Execution")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(
                "🚀 Run Main Pipeline", 
                disabled=st.session_state.pipeline_running,
                help="Run the complete DriveLM analysis pipeline"
            ):
                with st.spinner("Running pipeline... This may take several minutes."):
                    success = self.run_main_pipeline()
                    if success:
                        st.success("✅ Pipeline completed successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Pipeline failed. Check logs for details.")
        
        with col2:
            st.metric("Pipeline Status", 
                     "✅ Complete" if st.session_state.pipeline_executed 
                     else "⏳ Running" if st.session_state.pipeline_running 
                     else "🔴 Not Run")
        
        with col3:
            files_status = self.check_pipeline_files()
            files_ready = sum(files_status.values())
            st.metric("Files Generated", f"{files_ready}/{len(files_status)}")
        
        if st.session_state.pipeline_running or st.session_state.pipeline_output:
            st.subheader("📝 Pipeline Output")
            
            if st.session_state.pipeline_running:
                st.info("⏳ Pipeline is running... Output will appear below.")
            
            if st.session_state.pipeline_output:
                output_text = "\n".join(st.session_state.pipeline_output)
                st.text_area(
                    "Recent Log Output",
                    value=output_text,
                    height=300,
                    disabled=True
                )
                
                if st.session_state.pipeline_running:
                    time.sleep(2)
                    st.rerun()
        
        st.subheader("⚙️ Pipeline Configuration")
        
        config_info = {
            "NuScenes Path": "../data/nusccens",
            "DriveLM Path": "../data/drivelm_data/train_sample.json", 
            "Output Directory": "results", 
            "RAG Enhancement": "Enabled",
            "Log Level": "INFO"
        }
        
        for key, value in config_info.items():
            st.write(f"**{key}**: `{value}`")
    
    def load_json_file(self, filename: str) -> Optional[Dict]:
        try:
            file_path = self.output_dir / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return None
    
    def load_text_file(self, filename: str) -> Optional[str]:
        try:
            file_path = self.output_dir / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            return None
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return None
    
    def render_data_analysis_tab(self):
        st.header("📊 Data Analysis")
        
        if not st.session_state.pipeline_executed:
            st.warning("⚠️ Please run the pipeline first to generate analysis files.")
            return
        
        findings = self.load_text_file("findings.md")
        if findings:
            st.subheader("📝 Analysis Findings")
            st.markdown(findings)
        else:
            st.warning("Findings report not found.")
        
        analysis_results = self.load_json_file("analysis_results.json")
        if analysis_results:
            st.subheader("📈 Key Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            quality_metrics = analysis_results.get('quality_metrics', {})
            scene_stats = analysis_results.get('scene_statistics', {})
            
            with col1:
                st.metric("Total Questions", quality_metrics.get('total_questions', 'N/A'))
            with col2:
                st.metric("Avg Question Length", f"{quality_metrics.get('avg_question_length', 0):.1f} words")
            with col3:
                st.metric("Questions with Reasoning", quality_metrics.get('questions_with_reasoning', 'N/A'))
            with col4:
                st.metric("Unique Scenes", scene_stats.get('total_scenes', 'N/A'))
        
        rag_stats = self.load_json_file("unified_rag_enhanced_statistics.json")
        if rag_stats:
            st.subheader("🤖 RAG Enhancement Overview")
            
            overview = rag_stats.get('dataset_overview', {})
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Items with Scene Data", overview.get('items_with_scene_data', 0))
            with col2:
                st.metric("Items with Images", overview.get('items_with_images', 0))
            with col3:
                st.metric("Unique Scenes", overview.get('unique_scenes', 0))
    
    def render_dashboard_tab(self):
        st.header("📈 Interactive Dashboard")
        
        if not st.session_state.pipeline_executed:
            st.warning("⚠️ Please run the pipeline first to generate visualization data.")
            return
        
        analysis_results = self.load_json_file("analysis_results.json")
        if not analysis_results:
            st.error("Analysis results not found.")
            return
        
        st.subheader("🔍 Question Type Analysis")
        question_dist = analysis_results.get('question_type_distribution', {})
        
        if question_dist:
            col1, col2 = st.columns(2)
            
            with col1:
                df_bar = pd.DataFrame(
                    list(question_dist.items()),
                    columns=['Type', 'Count']
                )
                fig_bar = px.bar(df_bar, x='Type', y='Count', title="Question Types Distribution")
                fig_bar.update_layout(xaxis_tickangle=45)  # ✅ Correct syntax
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                fig_pie = px.pie(df_bar, values='Count', names='Type', title="Question Types (Pie Chart)")
                st.plotly_chart(fig_pie, use_container_width=True)
        
        st.subheader("📊 Data Quality Dashboard")
        quality_metrics = analysis_results.get('quality_metrics', {})
        
        if quality_metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                avg_q_len = quality_metrics.get('avg_question_length', 0)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=avg_q_len,
                    title={'text': "Avg Question Length (words)"},
                    gauge={'axis': {'range': [None, 50]}, 'bar': {'color': "darkblue"}}
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                avg_a_len = quality_metrics.get('avg_answer_length', 0)
                fig_gauge2 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=avg_a_len,
                    title={'text': "Avg Answer Length (words)"},
                    gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "darkgreen"}}
                ))
                st.plotly_chart(fig_gauge2, use_container_width=True)
    
    
    def load_rag_system(self):
        try:
            from RAG.pipeline import RAGPipeline

            progress = st.progress(0)
            status = st.empty()

            with st.spinner("⚡ Initializing RAG system..."):
                status.text("Creating RAG pipeline instance...")
                pipeline = RAGPipeline(output_dir="results")
                progress.progress(20)

                status.text("Initializing components...")
                if not pipeline.initialize_components():
                    st.error("❌ Failed to initialize components")
                    return None
                progress.progress(40)

                status.text("Building vector index...")
                if not pipeline.build_vector_index(force_rebuild=False):
                    st.error("❌ Failed to build vector index")
                    return None
                progress.progress(60)

                status.text("Loading generation models...")
                if not pipeline.load_generation_models(load_vision=True):
                    st.error("❌ Failed to load generation models")
                    return None
                progress.progress(80)

                status.text("Running pipeline test...")
                test = pipeline.answer_question("Test question", top_k=1, display_images=False)
                if "answer" not in test:
                    st.error("❌ Pipeline test failed")
                    return None
                progress.progress(100)

            status.text("✅ RAG system ready!")
            st.success("🎉 RAG system loaded successfully")
            return pipeline

        except Exception as e:
            st.error(f"❌ Error loading RAG system: {e}")
            st.code(traceback.format_exc())
            return None

    def reset_rag_system(self):
        self.rag_pipeline = None
        st.session_state.rag_loaded = False
        st.session_state.finetuning_completed = False

    def run_finetuning(self):
        if not RAG_AVAILABLE:
            st.error("RAG system not available")
            return False
        
        if not st.session_state.pipeline_executed:
            st.error("Please run the pipeline first to generate training data")
            return False
        
        try:
            with st.spinner("Running LoRA fine-tuning... This will take several minutes."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Preparing training data...")
                progress_bar.progress(20)
                
                enhanced_path = self.output_dir / "unified_rag_enhanced.json"
                unified_path = self.output_dir / "unified.json"
                
                training_data_path = None
                if enhanced_path.exists():
                    training_data_path = str(enhanced_path)
                    status_text.text("Using RAG-enhanced data...")
                elif unified_path.exists():
                    training_data_path = str(unified_path)
                    status_text.text("Using unified data...")
                else:
                    st.error("No training data found")
                    return False
                
                progress_bar.progress(50)
                status_text.text("Starting LoRA fine-tuning...")
                
                success = run_finetuning_pipeline(
                    output_dir=str(self.output_dir),
                    unified_data_path=training_data_path,
                    max_samples=200,
                    num_epochs=1
                )
                
                if success:
                    progress_bar.progress(100)
                    status_text.text("Fine-tuning completed!")
                    st.session_state.finetuning_completed = True
                    time.sleep(1)
                    status_text.empty()
                    progress_bar.empty()
                    return True
                else:
                    st.error("Fine-tuning failed")
                    return False
                    
        except Exception as e:
            st.error(f"Error during fine-tuning: {e}")
            logger.error(f"Fine-tuning error: {traceback.format_exc()}")
            return False
    
    def render_rag_tab(self):
        st.header("🤖 RAG Question Answering")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Load RAG Pipeline", disabled=st.session_state.get("pipeline_ready", False)):
                pipeline = self.load_rag_system()
                if pipeline:
                    st.session_state.rag_pipeline = pipeline
                    st.session_state.pipeline_ready = True
                    st.session_state.rag_loaded = True   
                    st.success("✅ RAG system loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load RAG system")

        with col2:
            if st.session_state.get("pipeline_ready", False):
                st.success("🟢 RAG System Ready")
            else:
                st.warning("🔴 RAG System Not Ready")

        if st.session_state.get("pipeline_ready", False) and st.session_state.get("rag_pipeline"):
            st.markdown("---")
            st.subheader("💬 Ask a Question")

            sample_questions = [
                "Will the detected vehicles keep moving?",
                "What objects are visible in the scene?",
                "Is it safe to change lanes now?",
                "What should I do if there's a pedestrian crossing?",
            ]

            col1, col2 = st.columns([3, 1])
            with col1:
                question = st.text_input("Enter your question:", placeholder="Ask about driving...")
            with col2:
                selected = st.selectbox("Examples:", [""] + sample_questions)
                if selected:
                    question = selected

            with st.expander("🔧 Advanced Options"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    top_k = st.slider("Top K Results", 1, 10, 3)
                with col2:
                    strategy = st.selectbox("Retrieval Strategy", ["hybrid", "semantic", "keyword"])
                with col3:
                    use_vision = st.checkbox("Use Vision Model", value=True)

            if question and st.button("🎯 Get Answer", type="primary"):
                with st.spinner("🔍 Searching knowledge base and generating answer..."):
                    try:
                        result = st.session_state.rag_pipeline.answer_question(
                            question=question,
                            top_k=top_k,
                            use_vision=use_vision,
                            retrieval_strategy=strategy,
                            display_images=False,
                        )

                        st.markdown("---")
                        st.subheader("🤖 Generated Answer")

                        answer = result.get("answer", "No answer generated")
                        confidence = result.get("confidence", 0.0)

                        if confidence > 0.7:
                            st.success(f"**Answer:** {answer}")
                        elif confidence > 0.4:
                            st.info(f"**Answer:** {answer}")
                        else:
                            st.warning(f"**Answer:** {answer}")

                        # --- Metrics ---
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Confidence", f"{confidence:.1%}")
                        with col2:
                            st.metric("Sources", result.get("sources_used", 0))
                        with col3:
                            st.metric("Images", result.get("total_images", 0))
                        with col4:
                            st.metric("Method", result.get("generation_method", "N/A"))

                        # --- Contexts ---
                        contexts = result.get("retrieved_contexts", [])
                        if contexts:
                            st.subheader("📚 Retrieved Contexts")
                            for i, ctx in enumerate(contexts):
                                with st.expander(f"Context {i+1}"):
                                    st.markdown(f"**Q:** {ctx.get('question', 'N/A')}")
                                    st.markdown(f"**A:** {ctx.get('answer', 'N/A')}")
                                    st.markdown(f"**Type:** {ctx.get('question_type', 'N/A')}")
                                    st.markdown(f"**Intent:** {ctx.get('semantic_intent', 'N/A')}")
                                    st.markdown(f"**Entities:** {', '.join(ctx.get('entities', []))}")
                                    st.markdown(f"**Scene Token:** {ctx.get('scene_token', 'N/A')}")
                                    st.markdown(f"**Complexity Score:** {ctx.get('complexity_score', 0)}")

                                    # Full metadata
                                    if "metadata" in ctx and ctx["metadata"]:
                                        st.json(ctx["metadata"])

                                    # Show image if available
                                    if ctx.get("image_path"):
                                        base_path = (Path(__file__).parent / "../data/nusccens").resolve()
                                        img_path = Path(ctx["image_path"])
                                        if not img_path.is_absolute():
                                            img_path = base_path / img_path
                                        if img_path.exists():
                                            st.image(str(img_path), caption=f"Context {i+1} Image")
                                        else:
                                            st.error(f"❌ Image not found: {img_path}")

                        # --- Retrieved Images ---
                        images = result.get("images", [])
                        if images:
                            st.subheader("🖼️ Retrieved Images")
                            for img in images:
                                if img.get("path"):
                                    try:
                                        st.image(img["path"], caption=f"Image {img.get('rank', '')}")
                                    except Exception as e:
                                        st.error(f"Could not load {img['path']}: {e}")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.code(traceback.format_exc())
        else:
            st.info("👆 Click 'Load RAG Pipeline' to start!")


    def render_evaluation_tab(self):
        st.header("📊 RAG Evaluation")

        if not st.session_state.pipeline_executed:
            st.warning("⚠️ Please run the pipeline first to generate evaluation data.")
            return

        data_file = self.output_dir / "unified_rag_enhanced.json"
        if not data_file.exists():
            st.error("❌ No enhanced RAG dataset found for evaluation.")
            return

        try:
            evaluator = RAGEvaluator(data_file=str(data_file))
            results = evaluator.evaluate()  

            st.subheader("📈 Quantitative Metrics")
            cols = st.columns(4)
            cols[0].metric("Exact Match", f"{results['metrics']['avg_exact_match']:.2%}")
            cols[1].metric("F1 Score", f"{results['metrics']['avg_f1']:.2%}")
            cols[2].metric("ROUGE-L", f"{results['metrics']['avg_rougeL']:.2%}")
            cols[3].metric("Semantic Sim.", f"{results['metrics']['avg_semantic_sim']:.2%}")

            st.subheader("🎨 Qualitative Evaluation & Visualization")

            if "results" in results:
                df = pd.DataFrame(results["results"])

                vis = EvaluationVisualizer(df)

                st.write("#### Semantic Similarity Distribution")
                st.pyplot(vis.plot_similarity_distribution(results["results"]))

                st.write("#### Answer Quality Distribution")
                st.pyplot(vis.plot_quality_distribution(results["results"]))

                if "confidence" in df.columns:
                    st.write("#### Confidence Distribution")
                    st.pyplot(vis.plot_confidence_distribution())

                # Metrics bar chart
                st.write("#### Metrics Overview")
                st.pyplot(vis.plot_metrics_bar(results["metrics"]))

            # Show qualitative examples
            st.write("#### 🔍 Sample Cases")
            for i, case in enumerate(results.get("results", [])[:5], 1):
                with st.expander(f"Example {i}"):
                    st.write(f"**Question:** {case['question']}")
                    st.write(f"**Predicted:** {case['pred_answer']}")
                    st.write(f"**Ground Truth:** {case['gold_answer']}")
                    st.json({
                        "Exact Match": case.get("exact_match"),
                        "F1": case.get("f1"),
                        "ROUGE-L": case.get("rougeL"),
                        "Semantic Sim": case.get("semantic_sim")
                    })

        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    
    def run(self):
        """Run the main application."""
        # Render sidebar
        self.render_sidebar()
        
        # Main title
        st.title("🚗 DriveLM Dataset Analysis & RAG System")
        st.markdown("### Integrated Pipeline Execution and Analysis Interface")
        st.markdown("---")
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔄 Pipeline",
            "📊 Data Analysis", 
            "📈 Dashboards",
            "🤖 RAG System",
            "📊 Evaluation"
        ])
        
        with tab1:
            self.render_pipeline_tab()
        
        with tab2:
            self.render_data_analysis_tab()
        
        with tab3:
            self.render_dashboard_tab()
        
        with tab4:
            self.render_rag_tab()
        
        with tab5:
            self.render_evaluation_tab()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center'>
                <p>🚗 DriveLM Analysis Suite | Integrated Pipeline & RAG System | 
                Built with Streamlit</p>
            </div>
            """,
            unsafe_allow_html=True
        )


def main():
    app = DriveLMIntegratedApp()
    app.run()


if __name__ == "__main__":
    main()