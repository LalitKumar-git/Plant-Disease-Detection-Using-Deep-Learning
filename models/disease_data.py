
DISEASE_INFO = {
    "late blight": {
        "symptoms": (
            "Late blight can cause water-soaked, dark green to brown "
            "lesions on leaves. In humid conditions, white fungal-like "
            "growth may appear on the underside of affected leaves. "
            "The disease can spread rapidly in favourable conditions."
        ),
        "treatment_chemical": (
            "Remove heavily infected plant material. Consult a local "
            "agricultural expert about an approved fungicide for your "
            "crop and region. Always follow the product label."
        ),
        "treatment_organic": (
            "Remove infected leaves, avoid overhead irrigation, improve "
            "air circulation and avoid working with wet plants."
        )
    }
}


def get_disease_info(predicted_class):
    crop, disease = predicted_class.split("___", 1)

    disease_name = disease.replace("_", " ").strip()
    disease_key = disease_name.lower()

    if "healthy" in disease_key:
        return {
            "symptoms": (
                "No disease detected. The plant is classified as healthy "
                "by the model."
            ),
            "treatment_chemical": (
                "No disease-specific chemical treatment is indicated."
            ),
            "treatment_organic": (
                "Continue regular plant care, proper watering and "
                "periodic monitoring."
            )
        }

    if disease_key in DISEASE_INFO:
        return DISEASE_INFO[disease_key]

    return {
        "symptoms": (
            f"Possible symptoms of {disease_name} require further "
            "verification. Please consult an agricultural expert."
        ),
        "treatment_chemical": (
            "Consult a qualified agricultural expert for a crop-specific, "
            "locally approved treatment. Follow the product label."
        ),
        "treatment_organic": (
            "Remove visibly infected plant material where appropriate, "
            "maintain good air circulation and avoid excessive moisture."
        )
    }
