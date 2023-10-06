import streamlit as st 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.utils import draw_bounding_boxes

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights,box_score_thresh = 0.8)
    model.eval()
    return model
model = load_model()

def make_prediction(img):
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction
def create_bounding_boxes(img,prediction):
    img_tensor = torch.tensor(img)
    bboxes = draw_bounding_boxes(img_tensor,boxes=prediction["boxes"],labels=prediction["labels"],colors = ["red" if label=="ambulance" else "green" for label in prediction["labels"]],width = 2)
    bboxes_np = bboxes.detach().numpy().transpose(1,2,0)
    return bboxes_np
bg_image = ""
st.markdown(bg_image,unsafe_allow_html=True)
st.title("Welcome to Hexor!")
upload = st.file_uploader(label = "Upload",type = ["png","jpg","jpeg"])
if upload:
    img = Image.open(upload)
    prediction = make_prediction(img)
    img_with_bbox = create_bounding_boxes(np.array(img).transpose(2,0,1),prediction)
    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(111)
    plt.imshow(img_with_bbox)
    plt.xticks([],[])
    plt.yticks([],[])
    ax.spines[["top","bottom","right","left"]].set_visible(False)

    st.pyplot(fig,use_container_width=True)
    del prediction["boxes"]
    st.header("Predicted Probabilities")
    st.write(prediction)


