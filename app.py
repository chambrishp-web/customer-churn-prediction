import textwrap
import streamlit as st
from app_utils import extract_text_from_pdf, clean_resume_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from fpdf import FPDF

# Hiring score thresholds
HIGH_SCORE = 85
GOOD_SCORE = 70
MEDIUM_SCORE = 50

# Comprehensive technical skills dictionary mapped to categories for generation
CORE_SKILLS_BANK = [
    "python", "r", "sql", "nosql", "mysql", "postgresql", "mongodb", "plsql",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data analytics", "data analysis", "statistics", "bi",
    "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch",
    "seaborn", "matplotlib", "tableau", "power bi", "excel", "sas", "spss",
    "html", "css", "javascript", "react", "angular", "node", "express", "django",
    "flask", "fastapi", "java", "c++", "c#", "php", "ruby", "aws", "azure", "gcp",
    "docker", "kubernetes", "jenkins", "git", "github", "ci/cd", "linux", "cloud"
]

SKILL_DETAILS_MAP = {
    "python": "core syntax, object-oriented programming, data structures, and automation scripts",
    "sql": "joins, subqueries, indexing, window functions, and database schema design",
    "pandas": "dataframes, data cleaning, filtering, merging, and handling missing values",
    "numpy": "multi-dimensional arrays, mathematical vector calculations, and linear algebra",
    "scikit-learn": "model training, pipelines, cross-validation, and performance metrics evaluation",
    "sklearn": "model training, pipelines, cross-validation, and performance metrics evaluation",
    "machine learning": "regression, classification, clustering, hyperparameter tuning, and overfitting prevention",
    "statistics": "mean, median, standard deviation, probability distributions, and hypothesis testing",
    "tableau": "interactive dashboards, data blending, calculations, and visual storytelling",
    "power bi": "DAX queries, data modeling, dashboard deployment, and ETL data transformations",
    "excel": "vlookups, index-match formulas, pivot tables, and macro charts processing",
    "aws": "EC2 compute compute instances, S3 storage buckets, IAM security access, and lambda architecture",
    "docker": "containerization strategies, writing Dockerfiles, images distribution, and isolated volumes"
}

def analyze_resume_vs_jd(jd_text, resume_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
    cosine_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    jd_skills = {skill for skill in CORE_SKILLS_BANK if skill in jd_text}
    resume_skills = {skill for skill in CORE_SKILLS_BANK if skill in resume_text}
    
    matching_skills = jd_skills.intersection(resume_skills)
    missing_skills = sorted(list(jd_skills.difference(resume_skills)))
    
    if len(jd_skills) > 0:
        skill_score = len(matching_skills) / len(jd_skills)
    else:
        skill_score = cosine_score 

    final_score = (0.60 * cosine_score) + (0.40 * skill_score)
    match_percentage = round(final_score * 100, 2)
    
    return match_percentage, sorted(list(matching_skills)), missing_skills

def rule_based_classifier(cleaned_text):
    scores = {
        "Data Science / Analytics": ["python", "machine learning", "data", "sql", "pandas", "scikit", "analytics", "tableau", "visualization"],
        "Web Development": ["javascript", "react", "html", "css", "nodejs", "web", "frontend", "backend", "django", "flask"],
        "DevOps / Cloud": ["aws", "docker", "kubernetes", "jenkins", "cicd", "cloud", "linux", "terraform", "azure"],
        "Human Resources": ["recruitment", "sourcing", "payroll", "hr", "talent", "onboarding", "employee", "management"]
    }
    max_matches = 0
    matched_category = "General / Other Profile"
    for category, keywords in scores.items():
        matches = sum(1 for word in keywords if word in cleaned_text)
        if matches > max_matches:
            max_matches = matches
            matched_category = category
    return matched_category

# UI Configuration with Sidebar and custom branding
st.set_page_config(page_title="Enterprise ATS Screener", layout="wide", initial_sidebar_state="expanded")
with st.sidebar:
    st.title("📄 AI Resume Screener")
    st.caption("Enterprise Edition v1.0")

    st.markdown("---")

    st.write("🏠 Dashboard")
    st.write("📊 ATS Score")
    st.write("✅ Matching Skills")
    st.write("❌ Missing Skills")
    st.write("💼 Hiring Recommendation")
    st.write("📥 Download Report")

    st.markdown("---")

    st.subheader("👨‍💻 Developer")
    st.write("Chambrish Prabhu")
    st.caption("Python | Machine Learning | Data Science")

# Custom CSS styling for premium look and feel
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; color: #1E3A8A; }
    .card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.image("logo.png", width=300)

st.title("📄 AI Resume Screener & ATS Matcher")
st.write(
    "Compare resumes with job descriptions using NLP and Machine Learning to generate ATS scores, skill gap analysis, and hiring recommendations."
)

# Main Layout structure
col_input, col_viz = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📥 Input Diagnostics Data")
    job_description = st.text_area("Paste Target Job Description (JD) Here", height=200)
    uploaded_file = st.file_uploader("Upload Applicant Resume (PDF only)", type=["pdf"])

if uploaded_file is not None and job_description.strip() != "":
    with col_viz:
        with st.spinner("Compiling multi-dimensional graphics..."):
            raw_resume_text = extract_text_from_pdf(uploaded_file)
            cleaned_resume = clean_resume_text(raw_resume_text)
            cleaned_jd = clean_resume_text(job_description)
            
            profile_category = rule_based_classifier(cleaned_resume)
            match_score, matching, missing = analyze_resume_vs_jd(cleaned_jd, cleaned_resume)

            # ==========================
            # Hiring Recommendation
            # ==========================
            if match_score >= HIGH_SCORE:
                hiring_status = "🟢 Highly Recommended"
                hiring_reason = "Excellent ATS match with strong skill alignment."

            elif match_score >= GOOD_SCORE:
                hiring_status = "🟢 Recommended"
                hiring_reason = "Good ATS match. A few improvements can strengthen the profile."

            elif match_score >= MEDIUM_SCORE:
                hiring_status = "🟡 Consider After Skill Improvement"
                hiring_reason = "Moderate ATS match. Improve the missing skills."

            else:
                hiring_status = "🔴 Not Recommended"
                hiring_reason = "Low ATS match. Significant improvements are needed."

            st.success("Deep Diagnostics Execution Complete!")
            st.caption(f"{hiring_status} — {hiring_reason}")

            # ATS Score Cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="🎯 ATS Match Score",
                value=f"{match_score}%"
            )

        with col2:
            st.metric(
                label="✅ Matching Skills",
                value=len(matching)
            )

        with col3:
            st.metric(
                label="❌ Missing Skills",
                value=len(missing)
            )

        # 📊 FEATURE 1: Circular Gauge Chart Configuration
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=match_score,
            title={'text': "Overall ATS Compatibility Score", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#1E3A8A"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#FCA5A5'},
                    {'range': [40, 60], 'color': '#FDE68A'},
                    {'range': [60, 80], 'color': '#93C5FD'},
                    {'range': [80, 100], 'color': '#86EFAC'}
                ],
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.write("---")
    
    # Bottom Layout Columns split for Skill Metrics
    col_left, col_right = st.columns(2, gap="large")
    
    with col_left:
        # 📈 FEATURE 2: Horizontal Bar Chart comparison
        st.subheader("📊 Skill Density Distribution Summary")
        categories = ['Matching Skills', 'Missing Gaps']
        counts = [len(matching), len(missing)]
        
        fig_bar = go.Figure([go.Bar(
            x=counts, 
            y=categories, 
            orientation='h',
            marker_color=['#22C55E', '#EF4444'],
            text=counts,
            textposition='auto'
        )])
        fig_bar.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.subheader("💡 Dynamic Contextual Recommendations")
        st.info(f"👉 **Predicted Target Domain**: This candidate fits a **{profile_category}** core profile pathway.")
        
        # 🤖 FEATURE 6: Automated suggestion feedback engine logic
        if missing:
            st.warning(f"👉 **Strategic Priority**: Fortify technical capability in **{missing[0].title()}** immediately to raise matching values.")
        else:
            st.balloons()
            st.success("👉 **Profile Optimal**: The candidate matches all benchmark structural parameters.")

    with col_right:
        st.subheader("📑 Algorithmic Skill Auditing")
        
        # Displaying verified indicators
        tab_match, tab_miss = st.tabs(["✅ Verified Matching Competencies", "❌ Identified Technical Vacancies"])
        
        with tab_match:
            if matching:
                st.write(", ".join([f"`{m.title()}`" for m in matching]))
            else:
                st.write("No matching background terms observed.")
                
        with tab_miss:
            if missing:
                for skill in missing:
                    skill_title = skill.title()
                    detail = SKILL_DETAILS_MAP.get(skill, "industry deployment frameworks and contextual case models")
                    with st.expander(f"🛑 {skill_title}"):
                        st.write(f"🔹 **Learn**: Study structural systems involving {detail}.")
                        st.write(f"🔹 **Build**: Construct a high-performance portfolio project highlighting {skill_title}.")
                        st.write(f"🔹 **Optimize**: Declare {skill_title} explicitly on your engineering technical header profile.")
            else:
                st.success("All targeted keywords mapped!")

    # 📄 FEATURE 4: Production PDF Generator Compilation Module
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    
    # Create Title
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(200, 10, txt="ATS RESUME METRICS EXECUTIVE PORTFOLIO", ln=True, align="C")
    pdf.ln(10)
    
    # Core Data Mapping
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, txt=f"Candidate Profile Target Classification: {profile_category}", ln=True)
    pdf.cell(200, 10, txt=f"Calculated Weighted Matching Value: {match_score}%", ln=True)
    pdf.cell(200, 10, txt=f"Aggregated Overlapping Skill Counts: {len(matching)} found", ln=True)
    pdf.cell(200, 10, txt=f"Identified Tool Architecture Gaps: {len(missing)} missing", ln=True)
    pdf.ln(5)
    
    # Write matching skills
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(200, 10, txt="Verified Structural Competencies:", ln=True)
    pdf.set_font("Helvetica", size=11)
    effective_page_width = pdf.w - pdf.l_margin - pdf.r_margin
    matching_text = ", ".join([s.title() for s in matching]) if matching else "None"
    pdf.multi_cell(effective_page_width, 8, txt=textwrap.fill(matching_text, width=95))
    pdf.ln(5)
    
    # Write missing skill structures
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(200, 10, txt="Required Optimization Roadmap Actions:", ln=True)
    pdf.set_font("Helvetica", size=10)
    for index, skill in enumerate(missing, 1):
        detail = SKILL_DETAILS_MAP.get(skill, "standard engineering frameworks and deployment modules")
        line = f"{index}. {skill.title()}: Focus learning paths onto {detail}. Deploy a portfolio asset project explicitly highlighting this competency."
        pdf.multi_cell(effective_page_width, 6, txt=textwrap.fill(line, width=95))
    
    pdf_output = pdf.output(dest="S")
    if isinstance(pdf_output, str):
        pdf_output = pdf_output.encode("latin-1")

    # Centered Global Export Button Area
    st.write("---")
    st.subheader("🖨️ Document Distribution")
    st.download_button(
        label="📥 Download Secure Analytical PDF Evaluation Report",
        data=pdf_output,
        file_name=f"ATS_Evaluation_Report_{uploaded_file.name.replace('.pdf', '')}.pdf",
        mime="application/pdf"
    )
else:
    with col_viz:
        st.info("💡 Awaiting parameters. Paste a target job description and drop your resume file on the left canvas to generate full visuals.")
        st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        Built with ❤️ using Python • Streamlit • Scikit-learn • NLP
        <br>
        © 2026 Chambrish Prabhu
    </div>
    """,
    unsafe_allow_html=True
)
