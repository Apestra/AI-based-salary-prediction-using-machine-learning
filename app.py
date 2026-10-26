import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="SalarySense AI", page_icon="💼", layout="wide")

st.markdown("""
<style>
.stApp {
    background: #f3f6fc;
    color: #172033;
}
.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 1.8rem;
    border-radius: 20px;
    color: #ffffff;
    background: linear-gradient(120deg, #101d38, #28558a);
    margin-bottom: 1rem;
}
.hero h1, .hero p {
    color: #ffffff !important;
}
.stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp li {
    color: #172033;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dce4f0;
    border-radius: 15px;
    padding: 1rem;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #172033 !important;
}
section[data-testid="stSidebar"] {
    background: #e8eef8;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>💼 SalarySense AI</h1><p>Machine-learning salary estimates and five-year career scenarios</p></div>',unsafe_allow_html=True)
st.caption("Educational prototype. Estimates depend on the dataset and are not guaranteed salary offers.")

ALIASES={
"experience":["years_of_experience","years experience","experience","experience_years","years_of_exp","yoe"],
"education":["education_level","education level","education","qualification","degree","educationlevel"],
"role":["job_title","job title","job_role","job role","role","designation","jobtitle","position"],
"salary":["salary","annual_salary","annual salary","salary_in_usd","income","salaryusd"],
"age":["age","employee_age"],"city":["city","location_city","work_city"],
"industry":["industry","sector"],"company_size":["company_size","company size","size"],
"remote":["remote","work_mode","work mode","work_type"]}
def standardise(df):
    df=df.copy(); df.columns=[str(c).strip().lower().replace("-","_") for c in df.columns]
    rename={}
    for target,names in ALIASES.items():
        for c in df.columns:
            if c in names or c.replace("_"," ") in names:
                rename[c]=target; break
    return df.rename(columns=rename)
def demo_data():
    rng=np.random.default_rng(24)
    roles=["Software Engineer","Data Analyst","Data Scientist","ML Engineer","Product Manager","Business Analyst","Cybersecurity Analyst","Cloud Engineer","UX Designer","DevOps Engineer","QA Engineer","Project Manager","Accountant","Marketing Manager","HR Specialist","Sales Executive","Mechanical Engineer","Civil Engineer","Financial Analyst","Database Administrator","Network Engineer","Frontend Developer","Backend Developer","AI Research Engineer","Operations Analyst"]
    edus=["High School","Diploma","Bachelor's","Master's","PhD"]
    cities=["Bengaluru","Delhi","Mumbai","Hyderabad","Pune","Chennai","Gurugram","Noida","Chandigarh","Kolkata","Jaipur","Ahmedabad","Ludhiana","Remote"]
    edu_pay={"High School":0,"Diploma":90000,"Bachelor's":180000,"Master's":300000,"PhD":420000}
    role_pay={r:int(rng.integers(0,900000)) for r in roles}
    city_pay={"Bengaluru":180000,"Mumbai":220000,"Delhi":140000,"Hyderabad":120000,"Pune":100000,"Gurugram":150000,"Noida":80000,"Chandigarh":50000,"Ludhiana":20000}
    rows=[]
    for _ in range(1800):
        exp=float(rng.uniform(0,28)); age=int(np.clip(rng.normal(23+exp*1.5,4),18,65))
        edu=str(rng.choice(edus,p=[.10,.12,.42,.29,.07])); role=str(rng.choice(roles)); city=str(rng.choice(cities))
        salary=max(180000,320000+exp*115000+edu_pay[edu]+role_pay[role]*.65+city_pay.get(city,0)+(age-30)*1800+rng.normal(0,180000))
        rows.append({"experience":round(exp,1),"education":edu,"role":role,"salary":round(salary),"age":age,"city":city,
                     "industry":str(rng.choice(["Technology","Finance","Healthcare","Manufacturing","Retail","Consulting"])),
                     "company_size":str(rng.choice(["Startup","Small","Medium","Large"]))})
    return pd.DataFrame(rows)

st.sidebar.header("📂 Dataset")
uploaded=st.sidebar.file_uploader("Upload Kaggle salary CSV",type=["csv"])
if uploaded:
    try: data=standardise(pd.read_csv(uploaded)); source=f"Uploaded: {uploaded.name}"
    except Exception as e: st.error(f"Could not read CSV: {e}"); st.stop()
else:
    data=demo_data(); source="Synthetic demo data (1,800 illustrative rows; not real Kaggle observations)"
missing=[c for c in ["experience","education","role","salary"] if c not in data.columns]
if missing:
    st.error("Missing required columns: "+", ".join(missing)+". CSV needs experience, education, job role and numeric salary.")
    st.info("Rename the CSV headers to Years of Experience, Education Level, Job Title, Salary, or their common aliases."); st.stop()
optional=[c for c in ["age","city","industry","company_size","remote"] if c in data.columns]
data=data[["experience","education","role","salary"]+optional].copy()
data["experience"]=pd.to_numeric(data["experience"],errors="coerce"); data["salary"]=pd.to_numeric(data["salary"],errors="coerce")
if "age" in data: data["age"]=pd.to_numeric(data["age"],errors="coerce")
data=data.dropna(subset=["experience","education","role","salary"])
data=data[(data.experience>=0)&(data.salary>0)]
for c in ["education","role","city","industry","company_size","remote"]:
    if c in data: data[c]=data[c].fillna("Unknown").astype(str).str.strip()
if len(data)<20: st.error("Please upload at least 20 valid rows; 100+ is preferable."); st.stop()
features=["experience","education","role"]+optional
numeric=[c for c in ["experience","age"] if c in features]
categorical=[c for c in features if c not in numeric]
@st.cache_resource(show_spinner="Training and evaluating models...")
def train(frame,features,numeric,categorical):
    X=frame[features]; y=frame["salary"]; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    prep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
      ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),categorical)])
    models={"Linear Regression":Pipeline([("prep",prep),("model",LinearRegression())]),
      "Random Forest":Pipeline([("prep",prep),("model",RandomForestRegressor(n_estimators=220,min_samples_leaf=2,random_state=42,n_jobs=-1))])}
    fitted={}; rows=[]
    for name,m in models.items():
        m.fit(Xtr,ytr); pred=m.predict(Xte); fitted[name]=m
        rows.append({"Model":name,"MAE":mean_absolute_error(yte,pred),"RMSE":mean_squared_error(yte,pred)**.5,"R²":r2_score(yte,pred)})
    return fitted,pd.DataFrame(rows)
try: fitted,metrics=train(data,features,numeric,categorical)
except Exception as e: st.error(f"Training failed: {e}"); st.stop()
a,b,c,d=st.columns(4); a.metric("Usable records",f"{len(data):,}"); b.metric("Job roles",f"{data.role.nunique():,}"); c.metric("Median salary",f"{data.salary.median():,.0f}"); d.metric("Model features",len(features))
st.caption(f"Dataset: {source}. Salary values use the same currency and time unit as the uploaded CSV.")
with st.expander("Preview data and quality checks"):
    st.dataframe(data.head(20),use_container_width=True); st.dataframe(data[["experience","salary"]].describe().round(2))
t1,t2,t3=st.tabs(["✨ Salary Predictor","📊 Model Lab","🗂️ Dataset Insights"])
with t2:
    st.subheader("Test-set model performance")
    st.dataframe(metrics.style.format({"MAE":"{:,.0f}","RMSE":"{:,.0f}","R²":"{:.3f}"}),use_container_width=True)
    fig,ax=plt.subplots(figsize=(8,3.5)); metrics.set_index("Model")[["MAE","RMSE"]].plot(kind="bar",ax=ax); ax.set_ylabel("Error in salary units"); ax.tick_params(axis="x",rotation=0); st.pyplot(fig)
    st.info("Selected model (lowest test RMSE): "+str(metrics.sort_values("RMSE").iloc[0]["Model"]))
with t3:
    x,y=st.columns(2)
    with x:
        fig,ax=plt.subplots(); ax.hist(data.salary,bins=30); ax.set_xlabel("Salary"); ax.set_ylabel("Records"); st.pyplot(fig)
    with y:
        fig,ax=plt.subplots(); ax.scatter(data.experience,data.salary,alpha=.35); ax.set_xlabel("Years experience"); ax.set_ylabel("Salary"); st.pyplot(fig)
    top=data.role.value_counts().head(15).index
    med=data[data.role.isin(top)].groupby("role").salary.median().sort_values()
    fig,ax=plt.subplots(figsize=(9,5)); med.plot(kind="barh",ax=ax); ax.set_xlabel("Median salary by role"); st.pyplot(fig)
with t1:
    st.subheader("Employee profile")
    st.write("Extra fields influence the model only if those fields are present in the uploaded CSV.")
    with st.form("predict"):
        a,b,c=st.columns(3)
        with a: exp=st.number_input("Years of experience",0.0,50.0,3.0,.5); age=st.number_input("Age",18,80,26)
        with b: edu=st.selectbox("Education",sorted(data.education.unique())); role=st.selectbox("Job role",sorted(data.role.unique()))
        with c:
            city=st.selectbox("City",sorted(data.city.unique()) if "city" in data else ["Not in dataset"])
            industry=st.selectbox("Industry",sorted(data.industry.unique()) if "industry" in data else ["Not in dataset"])
        a,b=st.columns(2)
        with a: size=st.selectbox("Company size",sorted(data.company_size.unique()) if "company_size" in data else ["Not in dataset"])
        with b: remote=st.selectbox("Work mode",sorted(data.remote.unique()) if "remote" in data else ["Not in dataset"])
        growth=st.slider("Assumed annual salary growth (%)",0.0,20.0,6.0,.5)
        submitted=st.form_submit_button("Predict salary & update charts",type="primary",use_container_width=True)
    if submitted:
        row={"experience":exp,"education":edu,"role":role}
        for col,val in [("age",age),("city",city),("industry",industry),("company_size",size),("remote",remote)]:
            if col in features: row[col]=val
        best=str(metrics.sort_values("RMSE").iloc[0]["Model"])
        prediction=max(0,float(fitted[best].predict(pd.DataFrame([row],columns=features))[0]))
        st.session_state["prediction"]={"salary":prediction,"growth":growth,"role":role,"edu":edu,"exp":exp,"age":age,"city":city,"model":best}
    if "prediction" in st.session_state:
        p=st.session_state["prediction"]; salary=p["salary"]; vals=[salary*(1+p["growth"]/100)**yr for yr in range(6)]
        a,b,c=st.columns(3); a.metric("Estimated current salary",f"{salary:,.0f}"); b.metric("Salary in 5 years",f"{vals[-1]:,.0f}",f"{vals[-1]-salary:,.0f} projected increase"); c.metric("Model selected",p["model"])
        st.caption(f"{p['role']} · {p['edu']} · {p['exp']:g} years experience · age {p['age']} · {p['city']}")
        left,right=st.columns(2)
        with left:
            fig,ax=plt.subplots(figsize=(7,4)); ax.plot(range(6),vals,marker="o"); ax.set_xticks(range(6)); ax.set_xlabel("Years from now"); ax.set_ylabel("Projected salary"); ax.grid(alpha=.25); st.pyplot(fig)
        with right:
            st.dataframe(pd.DataFrame({"Year":[f"Year {i}" for i in range(6)],"Projected salary":[round(v) for v in vals]}).style.format({"Projected salary":"{:,.0f}"}),use_container_width=True,hide_index=True)
        st.caption(f"The five-year curve assumes a constant {p['growth']:.1f}% annual increase compounded yearly. It is a scenario, not a future ML prediction or guarantee.")
    else: st.info("Enter a profile and click Predict salary & update charts.")
st.markdown("---")
st.caption("Do not upload personally identifiable employee data. Historical salary data may contain bias. Do not use this educational prototype as the sole basis for pay decisions.")
