import sys
from pathlib import Path


# Add src/ to Python import path
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))


from integration.road_safety_analyzer import analyze_video


result = analyze_video(
    input_path="data/sample_videos/road_test.mp4",
    output_path="outputs/dashboard_test/analyzed.mp4"
)


print("\n================================")
print("ANALYZER TEST COMPLETE")
print("================================")

print(
    "Output:",
    result["output_video"]
)

print(
    "Risk level:",
    result["risk_level"]
)

print(
    "Risk probability:",
    f"{result['risk_percent']:.2f}%"
)

print(
    "Risk history points:",
    len(result["risk_history"])
)

print(
    "Scene:",
    result["scene"]
)

print(
    "Reasons:",
    result["reasons"]
)

print(
    "FPS:",
    f"{result['fps']:.2f}"
)

print(
    "Latency:",
    f"{result['latency_ms']:.2f} ms"
)