from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

# Allow overriding the model via env var but default to the notebook's model
model_name = os.getenv("GOOGLE_MODEL") or "gemini-flash-latest"
llm = ChatGoogleGenerativeAI(model=model_name)

class EvaluationSchema(BaseModel):

    feedback: str = Field(description='Detailed feedbackfor the essay')
    score: int = Field(description='Score out of 10', ge=0, le=10)

structured_model = llm.with_structured_output(EvaluationSchema)

essay = """India in the Age of AI
As the world enters a transformative era defined by artificial intelligence (AI), India stands at a critical juncture — one where it can either emerge as a global leader in AI innovation or risk falling behind in the technology race. The age of AI brings with it immense promise as well as unprecedented challenges, and how India navigates this landscape will shape its socio-economic and geopolitical future.

India's strengths in the AI domain are rooted in its vast pool of skilled engineers, a thriving IT industry, and a growing startup ecosystem. With over 5 million STEM graduates annually and a burgeoning base of AI researchers, India possesses the intellectual capital required to build cutting-edge AI systems. Institutions like IITs, IIITs, and IISc have begun fostering AI research, while private players such as TCS, Infosys, and Wipro are integrating AI into their global services. In 2020, the government launched the National AI Strategy (AI for All) with a focus on inclusive growth, aiming to leverage AI in healthcare, agriculture, education, and smart mobility.

One of the most promising applications of AI in India lies in agriculture, where predictive analytics can guide farmers on optimal sowing times, weather forecasts, and pest control. In healthcare, AI-powered diagnostics can help address India’s doctor-patient ratio crisis, particularly in rural areas. Educational platforms are increasingly using AI to personalize learning paths, while smart governance tools are helping improve public service delivery and fraud detection.

However, the path to AI-led growth is riddled with challenges. Chief among them is the digital divide. While metropolitan cities may embrace AI-driven solutions, rural India continues to struggle with basic internet access and digital literacy. The risk of job displacement due to automation also looms large, especially for low-skilled workers. Without effective skilling and re-skilling programs, AI could exacerbate existing socio-economic inequalities.

Another pressing concern is data privacy and ethics. As AI systems rely heavily on vast datasets, ensuring that personal data is used transparently and responsibly becomes vital. India is still shaping its data protection laws, and in the absence of a strong regulatory framework, AI systems may risk misuse or bias.

To harness AI responsibly, India must adopt a multi-stakeholder approach involving the government, academia, industry, and civil society. Policies should promote open datasets, encourage responsible innovation, and ensure ethical AI practices. There is also a need for international collaboration, particularly with countries leading in AI research, to gain strategic advantage and ensure interoperability in global systems.

India’s demographic dividend, when paired with responsible AI adoption, can unlock massive economic growth, improve governance, and uplift marginalized communities. But this vision will only materialize if AI is seen not merely as a tool for automation, but as an enabler of human-centered development.

In conclusion, India in the age of AI is a story in the making — one of opportunity, responsibility, and transformation. The decisions we make today will not just determine India’s AI trajectory, but also its future as an inclusive, equitable, and innovation-driven society."""

class UPSCState(TypedDict):

    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float


def evaluate_language(state: UPSCState):

    prompt = f"Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}"
    output = structured_model.invoke(prompt)

    return {'language_feedback': output.feedback, 'individual_scores': [output.score]}


def evaluate_analysis(state: UPSCState):

    prompt = f"Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}"
    output = structured_model.invoke(prompt)

    return {'analysis_feedback': output.feedback, 'individual_scores': [output.score]}


def evaluate_thought(state: UPSCState):

    prompt = f"Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}"
    output = structured_model.invoke(prompt)

    return {'clarity_feedback': output.feedback, 'individual_scores': [output.score]}


def final_evaluation(state: UPSCState):

    # summary feedback
    prompt = f"Based on the following feedbacks create a summarized feedback \n+ language feedback - {state['language_feedback']} \n+ depth of analysis feedback - {state['analysis_feedback']} \n+ clarity of thought feedback - {state['clarity_feedback']}"
    overall_feedback = llm.invoke(prompt).content

    # avg calculate
    avg_score = sum(state['individual_scores'])/len(state['individual_scores'])

    return {'overall_feedback': overall_feedback, 'avg_score': avg_score}


def build_and_run_workflow(essay_text: str):
    graph = StateGraph(UPSCState)

    graph.add_node('evaluate_language', evaluate_language)
    graph.add_node('evaluate_analysis', evaluate_analysis)
    graph.add_node('evaluate_thought', evaluate_thought)
    graph.add_node('final_evaluation', final_evaluation)

    # edges
    graph.add_edge(START, 'evaluate_language')
    graph.add_edge(START, 'evaluate_analysis')
    graph.add_edge(START, 'evaluate_thought')

    graph.add_edge('evaluate_language', 'final_evaluation')
    graph.add_edge('evaluate_analysis', 'final_evaluation')
    graph.add_edge('evaluate_thought', 'final_evaluation')

    graph.add_edge('final_evaluation', END)

    workflow = graph.compile()

    initial_state = {"essay": essay_text}
    try:
        result = workflow.invoke(initial_state)
        print("Workflow result:\n", result)
        return result
    except Exception as e:
        print('Workflow raised an exception:', type(e).__name__, str(e))
        print('If this is a model NotFound error, set the environment variable GOOGLE_MODEL to a supported model name or call ListModels in the Google GenAI client.')
        raise


if __name__ == '__main__':
    # Try to run as a Streamlit app if Streamlit is available; otherwise run the CLI behavior
    try:
        import streamlit as st
    except Exception:
        # No Streamlit: preserve existing behavior (run two sample evaluations)
        print("Running evaluation on 'essay' sample...")
        build_and_run_workflow(essay)

        essay2 = """India and AI Time

Now world change very fast because new tech call Artificial Intel… something (AI). India also want become big in this AI thing. If work hard, India can go top. But if no careful, India go back.

India have many good. We have smart student, many engine-ear, and good IT peoples. Big company like TCS, Infosys, Wipro already use AI. Government also do program “AI for All”. It want AI in farm, doctor place, school and transport.

In farm, AI help farmer know when to put seed, when rain come, how stop bug. In health, AI help doctor see sick early. In school, AI help student learn good. Government office use AI to find bad people and work fast.

But problem come also. First is many villager no have phone or internet. So AI not help them. Second, many people lose job because AI and machine do work. Poor people get more bad.

One more big problem is privacy. AI need big big data. Who take care? India still make data rule. If no strong rule, AI do bad.

India must all people together – govern, school, company and normal people. We teach AI and make sure AI not bad. Also talk to other country and learn from them.

If India use AI good way, we become strong, help poor and make better life. But if only rich use AI, and poor no get, then big bad thing happen.

So, in short, AI time in India have many hope and many danger. We must go right road. AI must help all people, not only some. Then India grow big and world say \"good job India\"."""

        print("\nRunning evaluation on 'essay2' sample...")
        build_and_run_workflow(essay2)
    else:
        # Streamlit UI: keep existing logic unchanged; the UI only calls into existing functions
        st.set_page_config(page_title="Essay Evaluator", layout="wide")
        st.title("Essay Evaluator")
        st.markdown("Enter or paste an essay below and click **Evaluate**. You can also upload a .txt file.")

        # Sidebar reserved for configuration (no action buttons)

        col1, col2 = st.columns([2,1])

        # Helper: render textual fields that may be plain strings, dicts, or list-of-dicts
        def render_text_value(value):
            # Plain string -> render markdown
            if isinstance(value, str):
                st.markdown(value)
                return

            # Dict with 'text' or 'type'/'text'
            if isinstance(value, dict):
                if 'text' in value and isinstance(value['text'], str):
                    st.markdown(value['text'])
                    return
                # fallback to printing the dict
                st.write(value)
                return

            # List: iterate and render items
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, dict) and 'text' in item and isinstance(item['text'], str):
                        st.markdown(item['text'])
                    elif isinstance(item, str):
                        st.markdown(item)
                    else:
                        st.write(item)
                return

            # Fallback
            st.write(value)

        # Helper to render result consistently across Evaluate and Samples
        def render_result(res, title="Result"):
            if isinstance(res, dict):
                # Summary
                if 'overall_feedback' in res or 'avg_score' in res:
                    st.subheader("Summary")
                    if 'overall_feedback' in res:
                        st.markdown("**Overall feedback:**")
                        render_text_value(res.get('overall_feedback'))
                    if 'avg_score' in res:
                        st.metric("Average Score", value=res.get('avg_score'))

                # Detailed feedback and individual scores as metrics
                with st.expander(title, expanded=True):
                    for k in ['language_feedback', 'analysis_feedback', 'clarity_feedback']:
                        if k in res:
                            st.markdown(f"**{k.replace('_',' ').title()}:**")
                            render_text_value(res[k])

                    if 'individual_scores' in res:
                        sc = res['individual_scores']
                        st.markdown("**Individual Scores:**")
                        if isinstance(sc, (list, tuple)) and len(sc) > 0:
                            cols = st.columns(len(sc))
                            for i, s in enumerate(sc):
                                try:
                                    cols[i].metric(label=f"Score {i+1}", value=s)
                                except Exception:
                                    cols[i].write(s)
                        else:
                            st.write(sc)
            else:
                st.write(res)
        with col1:
            uploaded = st.file_uploader("Upload essay (.txt)", type=["txt"])
            if uploaded is not None:
                try:
                    essay_text = uploaded.getvalue().decode("utf-8")
                except Exception:
                    essay_text = uploaded.getvalue().decode("latin-1")
                essay_input = st.text_area("Essay", value=essay_text, height=400)
            else:
                essay_input = st.text_area("Essay", value=essay, height=400)

            run_eval = st.button("Evaluate")
            if run_eval:
                with st.spinner("Running evaluation..."):
                    try:
                        result = build_and_run_workflow(essay_input)
                        st.success("Evaluation completed")
                        render_result(result, title="Evaluation Result")
                    except Exception as e:
                        st.error(f"Workflow raised an exception: {type(e).__name__}: {e}")

        