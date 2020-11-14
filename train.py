from utils import train_dataloader
from utils import train_df, validate_df, cross_entropy_loss
from model import face_mask_detector_cnn

LEARNING_RATE = 0.00001

def train_model():
    optimizer = Adam(face_mask_detector_cnn.parameters(), lr=LEARNING_RATE)
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_dataloader(train_df), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'], data['mask']
            labels = labels.flatten()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = cross_entropy_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

print('Finished Training')