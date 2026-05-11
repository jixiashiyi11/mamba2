import os
import json

root_dir = '/home/wmwanghkmu/ZYH/Mamba/ADer/data/mvtec'
meta_data = {"train": {}, "test": {}}

classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

for cls in classes:
    meta_data["train"][cls] = []
    meta_data["test"][cls] = []
    
    # 扫描训练集
    train_path = os.path.join(root_dir, cls, 'train')
    if os.path.exists(train_path):
        for specie in os.listdir(train_path):
            specie_path = os.path.join(train_path, specie)
            if os.path.isdir(specie_path):
                for f in os.listdir(specie_path):
                    if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                        meta_data["train"][cls].append({
                            "img_path": os.path.relpath(os.path.join(specie_path, f), root_dir),
                            "mask_path": "",  # 训练集不需要 mask
                            "cls_name": cls,
                            "specie_name": specie,
                            "anomaly": 0
                        })
                        
    # 扫描测试集
    test_path = os.path.join(root_dir, cls, 'test')
    if os.path.exists(test_path):
        for specie in os.listdir(test_path):
            specie_path = os.path.join(test_path, specie)
            if os.path.isdir(specie_path):
                for f in os.listdir(specie_path):
                    if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                        anomaly = 0 if specie == 'good' else 1
                        mask_path = ""
                        if anomaly == 1:
                            # MVTec 的标准 mask 命名规则
                            base = os.path.splitext(f)[0]
                            mask_path = os.path.join(cls, 'ground_truth', specie, base + '_mask.png')
                            
                        meta_data["test"][cls].append({
                            "img_path": os.path.relpath(os.path.join(specie_path, f), root_dir),
                            "mask_path": mask_path,
                            "cls_name": cls,
                            "specie_name": specie,
                            "anomaly": anomaly
                        })

out_file = os.path.join(root_dir, 'meta.json')
with open(out_file, 'w') as f:
    json.dump(meta_data, f, indent=4)

print("高级版 meta.json 已经重新生成！现在每张图片都有详细档案了！")
