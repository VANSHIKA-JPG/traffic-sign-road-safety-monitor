class RiskEngine:

    def __init__(
        self,
        medium_threshold=0.35,
        high_threshold=0.65
    ):
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold


    def assess(
        self,
        risk_probability,
        features=None
    ):
        """
        risk_probability:
            LSTM probability for class 1 (RISK)

        features:
            Optional dictionary containing the latest
            YOLO + U-Net perception features.
        """

        # ----------------------------------------------
        # BASE TEMPORAL RISK LEVEL
        # ----------------------------------------------

        if risk_probability >= self.high_threshold:
            level = "HIGH"

        elif risk_probability >= self.medium_threshold:
            level = "MEDIUM"

        else:
            level = "LOW"


        # ----------------------------------------------
        # EXPLANATION
        # ----------------------------------------------

        reasons = []

        if features is not None:

            if features.get(
                "on_road_pedestrian_count", 0
            ) > 0:

                reasons.append(
                    "Pedestrian detected on drivable area"
                )


            if features.get(
                "on_road_object_count", 0
            ) >= 3:

                reasons.append(
                    "Multiple objects detected on drivable area"
                )


            if features.get(
                "largest_object_area_ratio", 0
            ) >= 0.08:

                reasons.append(
                    "Large nearby road object detected"
                )


            if features.get(
                "heavy_vehicle_count", 0
            ) > 0:

                reasons.append(
                    "Heavy vehicle present in road scene"
                )


        # ----------------------------------------------
        # TEMPORAL MODEL EXPLANATION
        # ----------------------------------------------

        if risk_probability >= self.high_threshold:

            reasons.append(
                "Temporal model indicates high risk"
            )

        elif risk_probability >= self.medium_threshold:

            reasons.append(
                "Temporal model indicates elevated risk"
            )


        if not reasons:

            reasons.append(
                "No major hazard indicators detected"
            )


        return {
            "level": level,

            "risk_probability":
                float(risk_probability),

            "risk_percent":
                float(risk_probability * 100),

            "reasons": reasons
        }


# --------------------------------------------------
# SIMPLE TEST
# --------------------------------------------------

if __name__ == "__main__":

    engine = RiskEngine()


    sample_features = {

        "on_road_pedestrian_count": 1,

        "on_road_object_count": 4,

        "largest_object_area_ratio": 0.09,

        "heavy_vehicle_count": 0
    }


    result = engine.assess(
        risk_probability=0.78,
        features=sample_features
    )


    print("Risk level:", result["level"])

    print(
        "Risk probability:",
        f"{result['risk_percent']:.2f}%"
    )

    print("Reasons:")

    for reason in result["reasons"]:
        print("-", reason)