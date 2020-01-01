import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity


def generate_prototypes_tensor_lowerdim(data):
    '''
    data:  dict{'support_feature'[],'support_y'[],'query_feature'[],'query_y'[]}
    return: prototype_ids, prototype_features
    '''
    support_feature, support_y, query_x, query_y = data['support_feature'], data['support_y'], data['query_feature'], data['query_y']

    # get prototype ids and prototype_features
    prototype_ids = []
    prototype_features = []

    dict = {}
    for i in range(support_y.shape[0]):
        classId = support_y[i]
        # turn feature from [video_frames,dim] to [dim]
        video_feature = support_feature[i]
        # video_feature = np.mean(video_feature, axis=0)
        if classId not in dict.keys():
            dict[classId] = [video_feature]
        else:
            dict[classId].append(video_feature)
    # print(dict.keys())

    for classId in dict.keys():
        prototype_ids.append(classId)
        prototype_feature = np.array(dict[classId])
        prototype_feature = np.mean(prototype_feature, axis=0)
        prototype_features.append(prototype_feature)
    prototype_features = np.array(prototype_features)
    # print(prototype_ids,prototype_features.shape)
    # [0.0, 1.0, 2.0, 3.0, 4.0] (5, 2048)
    return (prototype_ids, prototype_features)


def one_shot_classifier_prototype_lowerdim(data):
    '''
    data: dict{'support_feature[],'support_y'[],'query_feature'[],'query_y'[]}
    return : predicted_y
    '''
    # get input
    support_feature, support_y, query_feature, query_y = data['support_feature'], data['support_y'], data['query_feature'], data['query_y']

    # get prototypes_ids and prototype_features
    prototype_ids, prototype_features = generate_prototypes_tensor_lowerdim(data)
    # print(prototype_ids,prototype_features.shape)

    # get distance
    query_features = []
    for i in range(query_y.shape[0]):
        query_feature = query_feature[i]
        # query_feature = np.mean(query_feature, axis=0)
        query_features.append(query_feature)
    query_features = np.array(query_features)

    distance = cdist(query_features, prototype_features,metric='euclidean')

    # get probability
    distance = torch.FloatTensor(distance)
    probability = torch.nn.functional.softmax(-distance)
    # print('probability:',probability,'query_y:',query_y)
    
    '''
    # get loss
    loss = 0
    for i in range(query_y.shape[0]):
        label = query_y[i]
        classid = prototype_ids.index(label)
        loss = loss + (- torch.log(probability[i][classid]))
    loss = loss / query_y.shape[0]
    # print('prototype loss:',loss)
    # loss.backward()
    '''

    # caculate accuracy
    label = query_y
    probability = probability.data.cpu().numpy()
    predicted_y = np.argmax(probability, axis=1)
    #accuracy = np.mean(lddabel == predicted_y)
    # print('P: ', 'label:', label, 'predicted_y:', predicted_y, 'accuracy:', accuracy, 'loss:', loss)

    #return loss, accuracy
    return predicted_y


class Classifier():
    def __init__(self,classifier='protonet'):
        self.classifier = classifier


    def predict(self, data_result):
        # data_result: dict
        # classifier_type:
        # 1. protoNet
        # 2. SVM
        # 3. KNN
        # 4. logistic regression
        # 5. classifier NN
        if (self.classifier == 'protonet'):
            predicted_y = one_shot_classifier_prototype_lowerdim(data_result)

        elif (self.classifier == 'SVM'):
            classifier_SVM = SVC(C=10) 
            classifier_SVM.fit(data_result['support_feature'], data_result['support_y'])
            predicted_y = classifier_SVM.predict(data_result['query_feature'])
        elif (self.classifier == 'KNN'):
            classifier_KNN = KNeighborsClassifier(n_neighbors= k_shot)
            classifier_KNN.fit(data_result['support_feature'],data_result['support_y'])
            predicted_y = classifier_KNN.predict(data_result['query_feature'])
        elif(self.classifier == 'cosine'):
            distance_cosine = cosine_similarity(data_result['query_feature'],data_result['support_feature'])
            predicted_y = np.argsort(-distance_cosine)
            predicted_y = predicted_y[:,0]
        else:
            print('classifier type error.')
        return predicted_y




if __name__=='__name__':
    myClassifier=classifier()
