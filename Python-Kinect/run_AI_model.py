import AI_model
from AI_model import *
import ray

state_path = "state_dict_binary-wudu-T20-allP-overlap1.pth"
num_class = 2
max_hop = 1
dilation = 1
num_node = 25
dropout = 0
edge, center = get_edge()
hop_dis = get_hop_distance(num_node, edge, max_hop=max_hop)
A = get_adjacency(hop_dis, center, num_node, max_hop, dilation)
A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
model = Model(in_channels=3, num_class=num_class, A=A, edge_importance_weighting=True, dropout=dropout)
model.load_state_dict(torch.load(state_path, map_location=torch.device('cpu')))

features_test_npy = np.load("sample_data.npy")
print(features_test_npy.shape)

datapoint = features_test_npy
ray.shutdown()
ray.init(num_cpus=1, num_gpus=0)

@ray.remote
def parallelModel(pred_i):
    model_output = model(model_input[:, :, pred_i:pred_i + 20, :, :])
    state_output = F.log_softmax(model_output, dim=1).max(1)[1]
    #     probs = softmax(model_output) # Can be used to monitor the probabilities
    return state_output.item()


tt = time.time()
datapoint = features_test_npy
softmax = nn.Softmax(dim=1)
next_n_frames = 3
pred = Predictor(n=next_n_frames - 1)  # Loading motion prediction model
zeros = np.zeros((1, 3, next_n_frames - 1, 25, 1))
# datapoint = np.random.rand(1, 10)
# datapoint[0, 0:9] = datapoint[0, 1:10]
# datapoint[0, 10] = 0

# datapoint = np.random.rand(1, 3, 20, 25, 1)
# new_data = np.random.rand(3, 25)
# datapoint[:, :, 0:19, :, :] = datapoint[0, 1:10]
while True:
    tt = time.time()
    with torch.no_grad():
        # datapoint.shape = (1 batch size, 3 dims, 20 frames, 25 joints, 1 objects)
        model_input = pred.transform(np.append(datapoint, zeros, axis=2))
        # Shape = (1 batch size, 3 dims, 22 frames, 25 joints, 1 objects)
        model_input = torch.FloatTensor(pre_normalization(model_input))
        # state_output_ray = [parallelModel.remote(pred_i) for pred_i in range(1)]
        # state_output_ray = [parallelModel.remote(pred_i) for pred_i in range(next_n_frames)]
        # state_output = ray.get(state_output_ray)  # output example: [1, 1, 0]
        model_output = model(model_input[:, :, 0:0 + 20, :, :])
        state_output = F.log_softmax(model_output, dim=1).max(1)[1]
    # print(state_output)
    print(time.time()-tt)

