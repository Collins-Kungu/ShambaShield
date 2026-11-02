"""SmartMaize Platform - web demo (Streamlit). Everything runs inside this one app: no separate backend.

Run locally:   python -m streamlit run app.py
Deploy:        Streamlit Community Cloud, main file = app.py (see README.md)
"""

import pandas as pd
import streamlit as st
from PIL import Image

import credit_score
import maize_price_trend
import ood_gate
import pest_model
import rag_chatbot
import weather_alert_service

st.set_page_config(page_title="SmartMaize Platform", page_icon="🌽", layout="wide")

L = {
    "en": {
        "menu": "Services menu", "intro": "Choose a service:",
        "pages": {"pest": "🔬 Maize Pest Classifier", "weather": "⛈️ Severe Weather Alerts",
                  "price": "📈 Maize Price Trend", "chat": "💬 Chatbot & RAG Advisor",
                  "credit": "💳 Farmer Credit Score"},
        "pest_sub": "Identify fall armyworm damage on maize leaves with your trained MobileNetV2 model.",
        "pest_scope": "This tool only checks for fall armyworm damage on maize. It cannot detect other pests or diseases.",
        "src": "Image source", "cam_opt": "Take a photo", "up_opt": "Upload",
        "cam": "Photograph a maize leaf", "up": "Upload a maize leaf image (JPG/PNG)",
        "caption": "Your image", "analysing": "Running image analysis...", "loading": "Loading the model (first time only)...",
        "outcome": "Classification outcome",
        "healthy": "The maize plant appears to be **Healthy** (confidence: {c:.1f}%)",
        "affected": "Warning: the maize plant shows signs of **Fall Armyworm damage** (confidence: {c:.1f}%)",
        "low": "Low confidence. Retake the photo in good light, close to the leaf.",
        "unrecognised": "**Unrecognised image.** This does not look like a maize leaf photo the system knows, so no pest classification was made. Retake the photo close to a single maize leaf, in good light, with the leaf filling most of the frame.",
        "ood_score": "Familiarity score: {s:.2f} (needs to be 0 or higher)",
        "ood_off": "Image-familiarity check is unavailable, so the classifier ran without it. Details: {e}",
        "model_err": "Could not load the classifier. Place 'maize_pest_mobilenetv2.keras' in the 'model' folder. Details: {e}",
        "weather_sub": "Live 7-day forecast from Open-Meteo, turned into farm alerts.",
        "price_sub": "Real WFP market prices for Kenya (monthly, published with a delay).",
        "chat_sub": "Fall armyworm advice drawn from the sources you add.",
        "credit_sub": "Agricultural risk assessment for the micro-credit waitlist (demonstration logic).",
        "credit_note": "Demonstration scoring, not a validated credit model. Name and ID are not stored.",
        "demo": "Farmer details", "sci": "Field and risk parameters",
        "fid": "Farmer registration ID", "fname": "Farmer full name", "size": "Farm size (hectares)",
        "moist": "Soil moisture (%)", "ndvi": "NDVI (vegetation health, 0-1)", "risk": "Climate risk zone",
        "risk_opts": {"Low": "Low", "Medium": "Medium", "High": "High"},
        "run": "Run credit evaluation", "need_id": "Please provide both the farmer ID and name.",
        "report": "Evaluation report: {n} ({i})", "score": "Credit index score",
        "ok": "STATUS: ADDED TO LOAN WAITLIST", "hold": "STATUS: WAITLIST ON HOLD",
        "cols": ["Parameter", "Weight", "Input", "Points earned"], "breakdown": "Score breakdown",
        "ok_msg": "Congratulations {n}! Your farm metrics show a sustainable risk profile and you are queued for the ShambaShapeUp micro-credit waitlist.",
        "hold_msg": "The score of {s:.1f} is below the minimum of {t:.0f}. Advise the farmer on soil moisture conservation, fertilizer and improving crop health (NDVI) before re-applying.",
        "param": {"Soil moisture": "Soil moisture", "NDVI": "NDVI", "Farm size": "Farm size",
                  "Climate risk": "Climate risk", "Climate risk multiplier": "Climate risk multiplier"},
    },
    "sw": {
        "menu": "Menyu ya huduma", "intro": "Chagua huduma:",
        "pages": {"pest": "🔬 Ukaguzi wa wadudu wa mahindi", "weather": "⛈️ Tahadhari za hali ya hewa",
                  "price": "📈 Mwelekeo wa bei ya mahindi", "chat": "💬 Mshauri wa gumzo (RAG)",
                  "credit": "💳 Alama ya mkopo ya mkulima"},
        "pest_sub": "Tambua uharibifu wa viwavi jeshi vamizi kwenye majani ya mahindi kwa kutumia modeli yako ya MobileNetV2.",
        "pest_scope": "Zana hii hukagua uharibifu wa viwavi jeshi vamizi kwenye mahindi pekee. Haitambui wadudu au magonjwa mengine.",
        "src": "Chanzo cha picha", "cam_opt": "Piga picha", "up_opt": "Pakia",
        "cam": "Piga picha ya jani la mahindi", "up": "Pakia picha ya jani la mahindi (JPG/PNG)",
        "caption": "Picha yako", "analysing": "Inachanganua picha...", "loading": "Inapakia modeli (mara ya kwanza tu)...",
        "outcome": "Matokeo ya uchambuzi",
        "healthy": "Mmea wa mahindi unaonekana **mzima** (uhakika: {c:.1f}%)",
        "affected": "Onyo: mmea wa mahindi una dalili za **uharibifu wa viwavi jeshi vamizi** (uhakika: {c:.1f}%)",
        "low": "Uhakika ni mdogo. Piga picha tena kwenye mwanga mzuri, karibu na jani.",
        "unrecognised": "**Picha haitambuliki.** Haionekani kuwa picha ya jani la mahindi ambayo mfumo unaitambua, kwa hivyo hakuna uainishaji wa wadudu uliofanywa. Piga picha tena karibu na jani moja la mahindi, kwenye mwanga mzuri, jani likijaza sehemu kubwa ya picha.",
        "ood_score": "Alama ya uzoefu wa picha: {s:.2f} (inapaswa kuwa 0 au zaidi)",
        "ood_off": "Ukaguzi wa uzoefu wa picha haupatikani, kwa hivyo modeli ilifanya kazi bila huo. Maelezo: {e}",
        "model_err": "Imeshindwa kupakia modeli. Weka 'maize_pest_mobilenetv2.keras' kwenye folda ya 'model'. Maelezo: {e}",
        "weather_sub": "Utabiri wa siku 7 kutoka Open-Meteo, ukigeuzwa kuwa tahadhari za shamba.",
        "price_sub": "Bei halisi za soko za WFP nchini Kenya (kila mwezi, huchapishwa kwa kuchelewa).",
        "chat_sub": "Ushauri kuhusu viwavi jeshi vamizi kutoka kwenye vyanzo unavyoongeza.",
        "credit_sub": "Tathmini ya hatari ya kilimo kwa orodha ya kusubiri mikopo midogo (mantiki ya majaribio).",
        "credit_note": "Hii ni alama ya majaribio, si modeli ya mikopo iliyothibitishwa. Jina na nambari havihifadhiwi.",
        "demo": "Maelezo ya mkulima", "sci": "Vigezo vya shamba na hatari",
        "fid": "Nambari ya usajili wa mkulima", "fname": "Jina kamili la mkulima", "size": "Ukubwa wa shamba (hekta)",
        "moist": "Unyevu wa udongo (%)", "ndvi": "NDVI (afya ya mimea, 0-1)", "risk": "Eneo la hatari ya hali ya hewa",
        "risk_opts": {"Low": "Chini", "Medium": "Wastani", "High": "Juu"},
        "run": "Kokotoa alama ya mkopo", "need_id": "Tafadhali weka nambari na jina la mkulima.",
        "report": "Ripoti ya tathmini: {n} ({i})", "score": "Alama ya mkopo",
        "ok": "HALI: AMEWEKWA KWENYE ORODHA YA KUSUBIRI MKOPO", "hold": "HALI: ORODHA YA KUSUBIRI IMESIMAMISHWA",
        "cols": ["Kigezo", "Uzito", "Ingizo", "Pointi zilizopatikana"], "breakdown": "Mchanganuo wa alama",
        "ok_msg": "Hongera {n}! Vipimo vya shamba lako vinaonyesha hatari inayokubalika na umewekwa kwenye orodha ya kusubiri mikopo midogo ya ShambaShapeUp.",
        "hold_msg": "Alama {s:.1f} iko chini ya kiwango cha chini cha {t:.0f}. Mshauri mkulima kuhusu kuhifadhi unyevu wa udongo, mbolea na kuboresha afya ya mimea (NDVI) kabla ya kuomba tena.",
        "param": {"Soil moisture": "Unyevu wa udongo", "NDVI": "NDVI", "Farm size": "Ukubwa wa shamba",
                  "Climate risk": "Hatari ya hali ya hewa", "Climate risk multiplier": "Kigezo cha hatari ya hali ya hewa"},
    },
}

lang = st.sidebar.radio("Language / Lugha", ["en", "sw"], format_func={"en": "English", "sw": "Kiswahili"}.get)
t = L[lang]
st.sidebar.title("🌽 SmartMaize Platform")
page = st.sidebar.radio(t["menu"], list(t["pages"]), format_func=t["pages"].get)

# ----------------------------------------------------------------------------- 1. pest classifier
if page == "pest":
    st.title(t["pages"]["pest"])
    st.write(t["pest_sub"])
    st.caption(t["pest_scope"])
    mode = st.radio(t["src"], [t["cam_opt"], t["up_opt"]], horizontal=True)
    f = st.camera_input(t["cam"]) if mode == t["cam_opt"] else st.file_uploader(t["up"], type=["jpg", "jpeg", "png"])
    if f is not None:
        image = Image.open(f)
        st.image(image, caption=t["caption"], use_container_width=True)
        # Step 1: familiarity gate (One-Class SVM). If it is unavailable we fail open so the
        # classifier still works, and tell the user.
        gate = None
        try:
            if not ood_gate.is_loaded():
                with st.spinner(t["loading"]):
                    ood_gate.load()
            gate = ood_gate.check(image)
        except Exception as e:
            st.info(t["ood_off"].format(e=e))

        if gate is not None and not gate["familiar"]:
            # Step 2a: unfamiliar image -> do NOT pass it to MobileNetV2
            st.subheader(t["outcome"])
            st.warning(t["unrecognised"])
            st.caption(t["ood_score"].format(s=gate["score"]))
        else:
            # Step 2b: familiar image -> binary MobileNetV2 classifier
            try:
                if not pest_model.is_loaded():
                    with st.spinner(t["loading"]):
                        pest_model.load_model()
                with st.spinner(t["analysing"]):
                    res = pest_model.predict(image)
            except Exception as e:
                st.error(t["model_err"].format(e=e))
            else:
                st.subheader(t["outcome"])
                pct = res["confidence"] * 100
                if res["label"] == "healthy":
                    st.success(t["healthy"].format(c=pct))
                else:
                    st.error(t["affected"].format(c=pct))
                if res["low_confidence"]:
                    st.warning(t["low"])

# ----------------------------------------------------------------------------- 2. weather
elif page == "weather":
    st.title(t["pages"]["weather"])
    st.write(t["weather_sub"])
    weather_alert_service.render_weather_tab(lang)

# ----------------------------------------------------------------------------- 3. prices
elif page == "price":
    st.title(t["pages"]["price"])
    st.write(t["price_sub"])
    maize_price_trend.render_price_tab(lang)

# ----------------------------------------------------------------------------- 4. chatbot + RAG
elif page == "chat":
    st.title(t["pages"]["chat"])
    st.write(t["chat_sub"])
    rag_chatbot.render_rag_tab(lang)

# ----------------------------------------------------------------------------- 5. credit score
elif page == "credit":
    st.title(t["pages"]["credit"])
    st.write(t["credit_sub"])
    st.caption(t["credit_note"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(t["demo"])
        farmer_id = st.text_input(t["fid"], "F-9824")
        farmer_name = st.text_input(t["fname"], "Jane Atieno")
        farm_size = st.number_input(t["size"], min_value=0.1, max_value=100.0, value=2.5, step=0.1)
    with c2:
        st.subheader(t["sci"])
        soil = st.slider(t["moist"], 0.0, 100.0, 35.0, step=0.5)
        ndvi = st.slider(t["ndvi"], 0.0, 1.0, 0.65, step=0.01)
        zone = st.selectbox(t["risk"], list(t["risk_opts"]), format_func=t["risk_opts"].get)

    if st.button(t["run"]):
        if not farmer_id.strip() or not farmer_name.strip():
            st.error(t["need_id"])
        else:
            r = credit_score.evaluate_farmer(soil, ndvi, farm_size, zone)
            st.markdown("---")
            st.subheader(t["report"].format(n=farmer_name, i=farmer_id))
            m1, m2 = st.columns(2)
            m1.metric(t["score"], f"{r['score']:.1f} / 100")
            (m2.success if r["approved"] else m2.error)(t["ok"] if r["approved"] else t["hold"])
            st.markdown(f"### {t['breakdown']}")
            st.table(pd.DataFrame(
                [[t["param"].get(b["parameter"], b["parameter"]), b["weight"], b["input"],
                  f"{b['points']} / {b['max_points']}"] for b in r["breakdown"]], columns=t["cols"]))
            if r["approved"]:
                st.balloons()
                st.info(t["ok_msg"].format(n=farmer_name))
            else:
                st.warning(t["hold_msg"].format(s=r["score"], t=r["threshold"]))
