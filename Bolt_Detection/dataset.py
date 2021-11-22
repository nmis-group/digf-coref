from utils import *

# def collate_fn(batch):
#     return tuple(zip(*batch))

class BoltDataset(Dataset):

    def __init__(self, df, transform, split):
        super().__init__()
        self.split = split
        self.label = 'Bolt'
        if split == 'test':
            df = df[df['fold']==4]
            df = df.reset_index(drop=True)
        if split == 'valid':
            df = df[df['fold']==3] 
            df = df.reset_index(drop=True)
        if split == 'train':
            df = df[df['fold'].isin([0,1,2])]
            df = df.reset_index(drop=True)
        self.df = df
        print(len(self.df))
        self.transform = transform
    
    def get_image_and_labels_by_idx(self, index):
        path = self.df.filename[index]
        #print(path)
        bolts = self.df.total_bolts[index]
        img= cv2.imread(path)        
        img = img.transpose(1,0,2)
        img = np.flipud(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes = self.df.bboxes[index]
        #labels=['Bolt']*len(bboxes)
        #print(bboxes)
        return img, bboxes, bolts

    def __getitem__(self, index: int):
        
        (img, bboxes, bolts) = self.get_image_and_labels_by_idx(index)

        target = {}
        #print(bboxes)
        labels = torch.ones(len(bboxes), dtype=torch.int64)
        if self.transform is not None:
            sample = self.transform(**{
                    'image': img,
                    'bboxes': bboxes,
                    'labels': labels
                })
            transformed_image = sample['image']
            target['bboxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
            target['labels'] = torch.stack(tuple(sample['labels']))
        
        #print(target['bboxes'])
        #print(target['labels'])
        _, new_h, new_w = transformed_image.shape
        #transformed_bboxes = np.array(transformed_bboxes)
        #transformed_bboxes[:, [0, 1, 2, 3]] = transformed_bboxes[:, [1, 0, 3, 2]]  # convert to yxyx

        target["img_size"] = (new_h, new_w)
        target["img_scale"] = torch.tensor([1.0])

        return transformed_image, target, bolts
        

    def __len__(self) -> int:
        return len(self.df)
    
    def show_data(self, index: int):
        """Visualizes a single bounding box on the image"""
        (img, bboxes, bolts) = self.get_image_and_labels_by_idx(index)
        
        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=bboxes, labels=['Bolt']*len(bboxes))
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['labels']
        print(transformed_image.shape)
        img = transformed_image
        bboxes = transformed_bboxes
        img = img.cpu().permute(1,2,0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        for b in bboxes:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (100, 255, 200), 10)

            ((text_width, text_height), _) = cv2.getTextSize('Bolt', cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    

            #cv2.rectangle(img, (b[0], b[1] - int(1.3 * text_height)), (b[0] + text_width, b[1]), (255, 255, 255), -1)

            cv2.putText(
                img,
                text='Bolt',
                org=(int(b[0]), int(b[1]) - int(0.3 * text_height)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=3, 
                color=(255, 255, 255), 
                thickness = 8,
                lineType=cv2.LINE_AA,
            )
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        
        

        