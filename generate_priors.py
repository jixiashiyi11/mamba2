import torch
import open_clip

# 强制指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("1. 正在加载 BiomedCLIP...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

# 放入设备并冻结
model = model.to(device)
model.eval()

print("2. 正在准备全套多向语义弹药库 (Prompt Dictionary)...")
# 这里包含了完整无缺的 3 个器官的先验描述，千万不要修改或删减这里的结构
prompt_dict = {
    "brain": {
        "norm": "A normal healthy brain MRI scan with symmetric hemispheres and no visible lesions.",
        "abnorms": [
            "An abnormal brain MRI scan showing hyperintense tumor regions and peritumoral edema.",
            "A pathological brain scan exhibiting ischemic infarcts or hemorrhagic lesions.",
            "An abnormal brain medical image with significant structural asymmetry and mass effect."
        ]
    },
    "liver": {
        "norm": "A normal healthy liver CT scan with homogeneous parenchyma and clear boundaries.",
        "abnorms": [
            "An abnormal liver CT scan showing hypodense hepatic lesions and potential metastases.",
            "A pathological liver image exhibiting nodular surface and signs of cirrhosis or hepatocellular carcinoma.",
            "An abnormal liver scan containing fluid-filled cysts or atypical vascular hemangiomas."
        ]
    },
    "retinal": {
        "norm": "A normal healthy retinal fundus image with a clear optic disc and macula.",
        "abnorms": [
            "A retinal fundus image exhibiting microaneurysms and dot hemorrhages.",
            "An abnormal fundus scan showing visible hard exudates and cotton wool spots.",
            "A pathological retina image with signs of diabetic retinopathy or neovascularization."
        ]
    }
}

organ_deltas = {}

print("3. 正在提取多向语义向量并计算归一化的 Delta C...")
with torch.no_grad():
    for organ, prompts in prompt_dict.items():
        print(f"   -> 正在处理 {organ} ...")
        # Token 必须 .to(device)
        tokens_norm = tokenizer([prompts["norm"]]).to(device)
        c_norm = model.encode_text(tokens_norm)
        c_norm = c_norm / c_norm.norm(dim=-1, keepdim=True)

        tokens_abnorm = tokenizer(prompts["abnorms"]).to(device)
        c_abnorms = model.encode_text(tokens_abnorm)
        c_abnorms = c_abnorms / c_abnorms.norm(dim=-1, keepdim=True)

        # 计算差值
        delta_matrix = c_abnorms - c_norm

        # Delta 必须再次归一化！否则 FiLM 容易梯度爆炸
        delta_matrix = delta_matrix / delta_matrix.norm(dim=-1, keepdim=True)

        organ_deltas[organ] = delta_matrix.cpu()

torch.save(organ_deltas, 'uni_medical_delta_priors.pt')
print("✅ 终极弹药库构建完成，Delta 尺度已完全镇压！请检查当前目录下是否有 uni_medical_delta_priors.pt 文件。")