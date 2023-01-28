import json


def collate_fn(batch):
    return tuple(zip(*batch))


def format_boxes(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    new_boxes = np.zeros_like(boxes)
    new_boxes[:, 0] = y1
    new_boxes[:, 1] = x1
    new_boxes[:, 2] = y2
    new_boxes[:, 3] = x2

    return new_boxes


def get_device(cuda):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda}')

    else:
        device = torch.device('cpu')

    return device


def save_json_file(save_list, save_path):
    with open(save_path, 'w') as file:
        json.dump(save_list, file, indent=4)

    return None
