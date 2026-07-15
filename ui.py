import streamlit as st
import torch
import torch.nn as nn
import json

from PIL import Image
from torchvision import transforms, datasets


# -------------------------
# Config
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


st.set_page_config(
    page_title="Persian Character Recognition",
    page_icon="ب",
    layout="centered"
)


# -------------------------
# Model Architecture
# -------------------------

class DeepModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256*8*8,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                256,
                num_classes
            )
        )


    def forward(self,x):

        x=self.features(x)

        x=self.classifier(x)

        return x



# -------------------------
# Load Classes
# -------------------------

dataset = datasets.ImageFolder(
    "dataset/train_data"
)

classes = dataset.classes



with open(
    "labels.json",
    "r",
    encoding="utf-8"
) as f:

    labels=json.load(f)



# -------------------------
# Load Model
# -------------------------

@st.cache_resource
def load_model():

    model=DeepModel(
        len(classes)
    )

    model.load_state_dict(
        torch.load(
            "Alpha_Recognation_Model.pth",
            map_location=device
        )
    )


    model.to(device)

    model.eval()

    return model



model=load_model()



# -------------------------
# Transform
# -------------------------

transform=transforms.Compose([

    transforms.Resize(
        (128,128)
    ),

    transforms.ToTensor()

])



# -------------------------
# UI
# -------------------------

st.title(
    "♾️Persian Character Recognition"
)

st.write(
    "Deep Learning based Persian letter recognition|تشخیص حروف فارسی مبتنی بر یادگیری عمیق"
)



uploaded_file = st.file_uploader(
    "Upload Persian character image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)



if uploaded_file:


    image=Image.open(
        uploaded_file
    ).convert("RGB")


    st.image(
        image,
        caption="Input Image",
        width=200
    )


    img=transform(image)

    img=img.unsqueeze(0)

    img=img.to(device)



    with torch.no_grad():

        output=model(img)


        probs=torch.softmax(
            output,
            dim=1
        )


        confidence,index=torch.max(
            probs,
            dim=1
        )



    index=index.item()

    confidence=confidence.item()*100


    class_id=classes[index]

    character=labels[class_id]



    # Position detection

    if "ـ" in character:

        if character.startswith("ـ"):

            position="Ending/پایان"

        elif character.endswith("ـ"):

            position="Beginning/آغاز"

        else:

            position="Middle/میانی"


    else:

        position="Stdlone/مستقل"



    st.success(
        "Prediction Complete"
    )


    col1,col2,col3=st.columns(3)


    with col1:

        st.metric(
            "Character",
            character
        )


    with col2:

        st.metric(
            "Position",
            position
        )


    with col3:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )