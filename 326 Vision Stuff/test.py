from groundingdino.util.inference import load_model
config_path = "GroundingDINO_SwinT_OGC.py"
weights_path = "groundingdino_swint_ogc.pth"
model = load_model(config_path, weights_path)
print("Grounding DINO model loaded successfully!")