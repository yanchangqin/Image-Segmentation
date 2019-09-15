import torch

def get_accuracy(SR,GT,threshold=0.5):
    # SR : Segmentation Result
    # GT : Ground Truth
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    print(corr)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc