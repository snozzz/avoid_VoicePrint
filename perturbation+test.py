import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
import os

# ----------------------------------
#  第一步: 初始化模型和参数
# ----------------------------------
print("正在加载预训练的声纹识别模型 (ECAPA-TDNN)...")
# 使用SpeechBrain加载一个在VoxCeleb数据集上预训练的SOTA模型
# 第一次运行时会自动下载模型
spk_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-tdnn", savedir="pretrained_models/spkrec-ecapa-tdnn")
print("模型加载完毕。")

# 攻击参数
# epsilon: 控制扰动的最大幅度 (L-infinity norm)。值越小，扰动越不易察觉，但攻击性可能减弱。
epsilon = 0.002
# alpha: PGD攻击中每一步的步长
alpha = 0.0002
# num_iter: 攻击的迭代次数
num_iter = 50

# ----------------------------------
#  第二步: 加载并预处理音频
# ----------------------------------
# ⚠️ 请将这里的路径替换为您自己的WAV音频文件路径
# 为了演示，我们先创建一个虚拟的音频文件。在实际使用中，请注释掉这部分。
if not os.path.exists("my_voice.wav"):
    print("未找到示例音频，正在创建一个虚拟音频文件 'my_voice.wav'...")
    sample_rate = 16000
    dummy_audio = torch.randn(1, sample_rate * 4) # 4秒长的随机噪声
    torchaudio.save("my_voice.wav", dummy_audio, sample_rate)
    print("虚拟音频已创建。请替换为您的真实WAV文件以获得有意义的结果。")

audio_file = "my_voice.wav" 
if not os.path.exists(audio_file):
    raise FileNotFoundError(f"错误: 请确保音频文件 '{audio_file}' 存在于当前目录。")

print(f"正在加载音频文件: {audio_file}")
original_audio, sample_rate = torchaudio.load(audio_file)

# 确保音频是单声道，16kHz采样率 (模型要求)
if original_audio.shape[0] > 1:
    original_audio = torch.mean(original_audio, dim=0, keepdim=True)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    original_audio = resampler(original_audio)

# 将音频张量移动到合适的设备 (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_audio = original_audio.to(device)
spk_model = spk_model.to(device)
print(f"当前使用的设备: {device}")


# ----------------------------------
#  第三步: 实现PGD攻击函数 核心逻辑 ✨
# ----------------------------------
def generate_adversarial_perturbation(model, audio, epsilon, alpha, num_iter):
    """
    使用PGD算法生成对抗性扰动，以降低声纹相似度。
    """
    model.eval() # 设置为评估模式
    
    # 1. 计算原始声纹嵌入作为目标"锚点"
    # detach() 是为了不计算原始嵌入的梯度
    original_embedding = model.encode_batch(audio).detach()
    
    # 2. 初始化扰动 delta 为0
    delta = torch.zeros_like(audio, requires_grad=True)
    
    print("\n--- 开始生成对抗性扰动 ---")
    for i in range(num_iter):
        # 3. 计算扰动后音频的嵌入
        perturbed_audio = audio + delta
        perturbed_embedding = model.encode_batch(perturbed_audio)
        
        # 4. 定义损失函数：我们希望最大化距离，等价于最小化相似度，或最小化负的距离。
        # 这里我们使用负的余弦相似度作为损失，最小化它就等于让向量方向尽可能相反。
        loss = -torch.nn.functional.cosine_similarity(perturbed_embedding, original_embedding).mean()

        # 5. 反向传播计算梯度
        model.zero_grad()
        loss.backward()
        
        # 6. PGD核心步骤：根据梯度更新扰动
        # 6.1. 获取梯度符号
        grad_sign = delta.grad.data.sign()
        # 6.2. 更新delta (梯度下降，但因为损失是负数，所以实际上是梯度上升)
        delta.data = delta.data - alpha * grad_sign
        
        # 6.3. 投影步骤: 保证扰动大小在epsilon范围内
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        
        # 6.4. 保证添加扰动后的音频值在有效范围内 (通常是[-1, 1])
        perturbed_audio.data = torch.clamp(perturbed_audio.data, -1.0, 1.0)
        # 必须更新delta，因为裁剪perturbed_audio后，delta也变了
        delta.data = perturbed_audio.data - audio.data

        # 清除梯度以便下次迭代
        delta.grad.data.zero_()

        if (i + 1) % 10 == 0:
            print(f"迭代 [{i+1}/{num_iter}], 当前损失 (负相似度): {loss.item():.4f}")

    print("--- 扰动生成完毕 ---")
    return delta.detach()


# ----------------------------------
#  第四步: 执行并验证结果
# ----------------------------------
# 生成扰动
perturbation = generate_adversarial_perturbation(spk_model, original_audio, epsilon, alpha, num_iter)

# 创建受保护的音频
adversarial_audio = original_audio + perturbation
# 再次确保值在-1到1之间
adversarial_audio = torch.clamp(adversarial_audio, -1.0, 1.0)


# 验证效果
print("\n--- 验证攻击效果 ---")
original_emb = spk_model.encode_batch(original_audio)
adversarial_emb = spk_model.encode_batch(adversarial_audio)

# 计算原始音频和受保护音频之间的声纹相似度
similarity_score = torch.nn.functional.cosine_similarity(original_emb, adversarial_emb).item()

print(f"原始声纹 vs. 受保护声纹 的相似度: {similarity_score:.4f}")
print("一个成功的攻击应该使这个分数显著降低 (理想情况接近0或负数)。")
print("如果分数为1.0，意味着攻击完全失败。接近0意味着攻击成功。")

# ----------------------------------
#  第五步: 保存结果
# ----------------------------------
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 保存文件
original_path = os.path.join(output_dir, "original.wav")
adv_path = os.path.join(output_dir, "adversarial_protected.wav")
pert_path = os.path.join(output_dir, "perturbation_noise.wav")

torchaudio.save(original_path, original_audio.cpu(), 16000)
torchaudio.save(adv_path, adversarial_audio.cpu(), 16000)
# 将扰动本身也保存为音频，可以听听它是什么样的 (通常是微弱的噪音)
torchaudio.save(pert_path, perturbation.cpu(), 16000)

print(f"\n处理完成！结果已保存至 '{output_dir}' 文件夹:")
print(f"  - 原始音频: {original_path}")
print(f"  - 受保护的音频: {adv_path}")
print(f"  - 生成的扰动噪音: {pert_path}")
print("\n您可以试着听一下'original.wav'和'adversarial_protected.wav'，看看听感上有多大差异。")
print("再听听'perturbation_noise.wav'，了解我们添加的“水印”是什么样的。")