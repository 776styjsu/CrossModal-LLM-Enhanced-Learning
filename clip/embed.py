import open_clip
import open_clip.tokenizer as tokenizer
import torch


def create_model_and_transforms(model_name: str = 'convnext_base_w',
                                pretrained: str = 'laion2b_s13b_b82k_augreg',
                                device='cuda'):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained)
    model = model.to(device)
    model.eval()
    print('Model loaded: {}'.format(model_name))
    print('Pretrained weights loaded: {}'.format(pretrained))
    return model, preprocess


def get_image_embed(image, model, preprocess, device='cuda'):
    image_input = preprocess(image).unsqueeze(0).to(device)
    image_features = model.encode_image(image_input).float()
    return image_features


def get_text_embed(tokenized_texts, model, device='cuda'):
    tokenized_texts = tokenized_texts.to(device)
    text_features = model.encode_text(tokenized_texts).float()
    return text_features

def get_cosine_similarity(image_features, text_features):
    image_features = image_features.squeeze(0)
    text_features = text_features.squeeze(0)
    cosine_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    similarity = cosine_similarity(image_features, text_features)
    return similarity
