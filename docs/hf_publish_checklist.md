# Hugging Face Publish Checklist

## Trước khi upload

- [ ] Checkpoint cuối cùng nằm trong `outputs/sft/.../final`
- [ ] Tokenizer files đã được save cùng checkpoint
- [ ] Bạn đã đăng nhập Hugging Face
- [ ] Repo id mong muốn đã chốt

## Nên upload cái gì?

### Option A: Upload adapter LoRA

Phù hợp khi:
- muốn nhẹ
- muốn public phần fine-tuning của bạn
- base model vẫn là `google/gemma-4-26B-A4B-it`

### Option B: Upload merged model

Phù hợp khi:
- muốn inference trực tiếp dễ hơn
- chấp nhận artifact lớn hơn

## Tên repo gợi ý

- `QuangVoAI/multimodal-empathy-mental-health-gemma26b-task1`
- `QuangVoAI/gemma26b-avamerg-esconv-task1`

## Nội dung model card nên có

1. **Model description**
2. **Base model**
3. **Training datasets**
4. **Evaluation benchmark**
5. **Limitations**
6. **Intended use**

## Lưu ý an toàn

Vì đây là bài toán mental health, model card nên ghi rõ:

- mô hình dùng cho **nghiên cứu**
- không thay thế chuyên gia sức khỏe tinh thần
- phản hồi của mô hình cần được xem xét thận trọng trong tình huống nguy cơ cao
