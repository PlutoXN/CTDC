
import os.path


from PrecessData_train import get_data,get_data_test

from ResourceRankConfidence_train import get_RRankConfidence
from TransConfidence_train import get_TransConfidence_train
from PrecessData_train import get_dict_entityRank
from torch.utils.data import DataLoader, Subset, TensorDataset
import pickle
from tqdm import tqdm  # 用于显示进度条

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn

# 1. 定义 Swish 激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 2. 定义 MultiHeadAttention 模块
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        return attn_output

# 3. 定义 Attention 机制
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / (self.v.size(0) ** 0.5)
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_size]
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, seq_len, hidden_size]
        energy = energy.transpose(2, 1)  # [batch_size, hidden_size, seq_len]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch_size, 1, hidden_size]
        energy = torch.bmm(v, energy)  # [batch_size, 1, seq_len]
        attn_weights = F.softmax(energy.squeeze(1), dim=1)  # [batch_size, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch_size, hidden_size]
        return context, attn_weights

# 4. 定义 EnhancedGRUCell
class EnhancedGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_steps=3, alpha=0.1, num_heads=4):
        super(EnhancedGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.alpha = alpha

        # 门控机制
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size + hidden_size, hidden_size)
        self.candidate_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # 层归一化
        self.reset_gate_norm = nn.LayerNorm(hidden_size)
        self.update_gate_norm = nn.LayerNorm(hidden_size)
        self.candidate_gate_norm = nn.LayerNorm(hidden_size)

        # 多头注意力机制
        self.multihead_attn = MultiHeadAttention(hidden_size, num_heads)

        # 动态时间步长门控
        self.time_gate = nn.Linear(hidden_size, 1)

        # 残差连接
        self.residual = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

    def forward(self, x, prev_h, prev_states):
        # 使用多头注意力机制对前n时刻的隐藏状态进行加权
        prev_states_t = prev_states.transpose(0, 1)  # [batch, n_steps, hidden_size]
        attn_output = self.multihead_attn(prev_h.unsqueeze(0), prev_states_t, prev_states_t)  # [1, batch, hidden_size]
        context_vector = attn_output.squeeze(0)  # [batch, hidden_size]

        # 动态时间步长门控
        time_weights = torch.sigmoid(self.time_gate(context_vector))  # [batch, 1]
        context_gate = self.alpha * context_vector * time_weights  # 控制其影响力

        # 拼接当前输入和前一隐藏状态
        combined = torch.cat([x, prev_h], dim=-1)  # [batch, input_size + hidden_size]

        # 更新门
        combined_with_context = torch.cat([combined, context_gate], dim=-1)  # [batch, input_size + 2*hidden_size]
        z_t = torch.sigmoid(self.update_gate(combined_with_context))  # [batch, hidden_size]
        z_t = self.update_gate_norm(z_t)

        # 重置门
        r_t = torch.sigmoid(self.reset_gate(combined))  # [batch, hidden_size]
        r_t = self.reset_gate_norm(r_t)

        # 重置的隐藏状态
        r_hidden = r_t * prev_h  # [batch, hidden_size]
        candidate_input = torch.cat([x, r_hidden], dim=-1)  # [batch, input_size + hidden_size]
        h_candidate = torch.tanh(self.candidate_gate(candidate_input))  # [batch, hidden_size]
        h_candidate = self.candidate_gate_norm(h_candidate)

        # 最终隐藏状态
        h_t = (1 - z_t) * prev_h + z_t * h_candidate  # [batch, hidden_size]

        # 残差连接
        h_t = h_t + self.residual(x)  # [batch, hidden_size]

        return h_t

# 5. 定义 EnhancedGRU（单向、多尺度）
class EnhancedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_steps=3, alpha=0.1, bidirectional=False, num_heads=4):
        super(EnhancedGRU, self).__init__()
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.bidirectional = bidirectional
        self.gru_cell_forward = EnhancedGRUCell(input_size, hidden_size, n_steps, alpha, num_heads)
        if self.bidirectional:
            self.gru_cell_backward = EnhancedGRUCell(input_size, hidden_size, n_steps, alpha, num_heads)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_size]
        """
        batch_size, seq_len, input_size = x.size()
        device = x.device

        # Forward direction
        h_t_forward = torch.zeros(batch_size, self.hidden_size).to(device)
        prev_states_forward = torch.zeros(batch_size, self.n_steps, self.hidden_size).to(device)
        outputs_forward = []
        for t in range(seq_len):
            h_t_forward = self.gru_cell_forward(x[:, t, :], h_t_forward, prev_states_forward)
            outputs_forward.append(h_t_forward.unsqueeze(1))  # [batch, 1, hidden_size]
            prev_states_forward = torch.cat([prev_states_forward[:, 1:, :], h_t_forward.unsqueeze(1)], dim=1)

        if self.bidirectional:
            # Backward direction
            h_t_backward = torch.zeros(batch_size, self.hidden_size).to(device)
            prev_states_backward = torch.zeros(batch_size, self.n_steps, self.hidden_size).to(device)
            outputs_backward = []
            for t in reversed(range(seq_len)):
                h_t_backward = self.gru_cell_backward(x[:, t, :], h_t_backward, prev_states_backward)
                outputs_backward.insert(0, h_t_backward.unsqueeze(1))  # [batch, 1, hidden_size]
                prev_states_backward = torch.cat([prev_states_backward[:, 1:, :], h_t_backward.unsqueeze(1)], dim=1)

            # Concatenate forward and backward outputs
            outputs_forward = torch.cat(outputs_forward, dim=1)  # [batch, seq_len, hidden_size]
            outputs_backward = torch.cat(outputs_backward, dim=1)  # [batch, seq_len, hidden_size]
            outputs = torch.cat([outputs_forward, outputs_backward], dim=-1)  # [batch, seq_len, hidden_size*2]
        else:
            # Only forward outputs
            outputs = torch.cat(outputs_forward, dim=1)  # [batch, seq_len, hidden_size]

        return outputs  # [batch, seq_len, hidden_size*2 if bidirectional else hidden_size]

# 6. 定义 PathGNN（图神经网络模块）
class PathGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super(PathGNN, self).__init__()
        self.gat1 = nn.Linear(input_dim, hidden_dim * num_heads)
        self.gat2 = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)

    def forward(self, x, edge_index=None):
        """
        x: [batch_size, seq_len, input_dim]
        edge_index: 图结构信息（如果有的话）
        """
        # 简单的GNN实现，实际应用中可以使用更复杂的GNN模块如GATConv
        x = F.elu(self.gat1(x))  # [batch, seq_len, hidden_dim * num_heads]
        x = self.gat2(x)  # [batch, seq_len, hidden_dim]
        return x

# 7. 定义 PathCNN（卷积神经网络模块）
class PathCNN(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size=3, padding=1):
        super(PathCNN, self).__init__()
        self.conv1d = nn.Conv1d(input_channels, out_channels, kernel_size, padding=padding)
        self.activation = Swish()
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        """
        x: [batch, seq_len, feature_dim]
        """
        x = x.transpose(1, 2)  # [batch, feature_dim, seq_len]
        x = self.conv1d(x)      # [batch, out_channels, new_seq_len]
        x = self.activation(x)
        x = self.pool(x)        # [batch, out_channels, new_seq_len//2]
        x = x.transpose(1, 2)   # [batch, new_seq_len//2, out_channels]
        return x

# 8. 定义最终模型（修改后的版本）
class create_Model_BiLSTM_BP(nn.Module):
    def __init__(self, entvocabsize, relvocabsize, ent2vec, rel2vec, input_path_length, ent_emd_dim, rel_emd_dim):
        super(create_Model_BiLSTM_BP, self).__init__()

        # 嵌入层
        self.ent_embedding = nn.Embedding(entvocabsize + 2, ent_emd_dim)
        self.rel_embedding = nn.Embedding(relvocabsize + 2, rel_emd_dim)

        # 加载预训练嵌入
        self.ent_embedding.weight.data.copy_(torch.tensor(ent2vec, dtype=torch.float32))
        self.rel_embedding.weight.data.copy_(torch.tensor(rel2vec, dtype=torch.float32))
        self.ent_embedding.weight.requires_grad = False  # 根据需求设置是否微调
        self.rel_embedding.weight.requires_grad = False

        # PathGNN 模块
        self.path_gnn = PathGNN(input_dim=ent_emd_dim * 3 + rel_emd_dim * 3, hidden_dim=100, num_heads=4)  # 增加hidden_dim和num_heads

        # PathCNN 模块
        self.path_cnn = PathCNN(input_channels=100, out_channels=100, kernel_size=3, padding=1)

        input_size = 100
        hidden_size = 100
        num_heads = 4
        n_steps = 5  #KG跳数

        # Enhanced GRU 模块（单向、多尺度）
        self.path_gru = EnhancedGRU(input_size=input_size, hidden_size=hidden_size, n_steps=n_steps, alpha=0.1, bidirectional=False, num_heads=num_heads)
        self.path_gru2 = EnhancedGRU(input_size=input_size, hidden_size=hidden_size, n_steps=n_steps, alpha=0.1, bidirectional=False, num_heads=num_heads)
        self.path_gru3 = EnhancedGRU(input_size=input_size, hidden_size=hidden_size, n_steps=n_steps, alpha=0.1, bidirectional=False, num_heads=num_heads)

        path_dim = 100
        rrank_dim = 100
        fin_dim = 100
        Dropout_rate = 0.5

        # 全连接层（调整至100和64）
        self.fc_path1 = nn.Sequential(
            nn.LayerNorm(path_dim),
            nn.Dropout(Dropout_rate),
            nn.Linear(path_dim, path_dim),
            Swish(),
            nn.Linear(path_dim, 1),  # 修正为 100 -> 1
            nn.Sigmoid()
        )

        self.fc_path2 = nn.Sequential(
            nn.LayerNorm(path_dim),
            nn.Dropout(Dropout_rate),
            nn.Linear(path_dim, path_dim),
            Swish(),
            nn.Linear(path_dim, 1),  # 修正为 100 -> 1
            nn.Sigmoid()
        )
        self.fc_path3 = nn.Sequential(
            nn.LayerNorm(path_dim),
            nn.Dropout(Dropout_rate),
            nn.Linear(path_dim, path_dim),
            Swish(),
            nn.Linear(path_dim, 1),  # 修正为 100 -> 1
            nn.Sigmoid()
        )

        # TransE 和 RRank 特征
        self.fc_rrank = nn.Sequential(
            nn.Linear(6, rrank_dim),
            nn.Tanh(),
            nn.Dropout(Dropout_rate),
            nn.Linear(rrank_dim, 1),
            nn.Sigmoid()
        )

        # 最终分类层（调整中间层大小）
        self.fc_bp = nn.Sequential(
            nn.Linear(5, fin_dim),
            Swish(),
            nn.Dropout(Dropout_rate),
            nn.Linear(fin_dim, rrank_dim),  # 修正为 100 -> 64
            Swish(),
            nn.Dropout(Dropout_rate),
            nn.Linear(rrank_dim, 2),
        )

    def forward(self, inputs):
        """
        inputs: 元组包含14个输入
        (ent_h, ent_t, rel_r,
         path_h, path_t, path_r,
         path2_h, path2_t, path2_r,
         path3_h, path3_t, path3_r,
         transE, rrank)
        """
        (ent_h, ent_t, rel_r,
         path_h, path_t, path_r,
         path2_h, path2_t, path2_r,
         path3_h, path3_t, path3_r,
         transE, rrank) = inputs

        # 嵌入
        ent_h_emb = self.ent_embedding(ent_h).squeeze(1)  # [batch_size, ent_emd_dim]
        ent_t_emb = self.ent_embedding(ent_t).squeeze(1)
        rel_r_emb = self.rel_embedding(rel_r).squeeze(1)

        # Path 1
        path1_h_emb = self.ent_embedding(path_h)  # [batch, path_length, ent_emd_dim]
        path1_t_emb = self.ent_embedding(path_t)
        path1_r_emb = self.rel_embedding(path_r)

        path1_input = torch.cat([
            ent_h_emb.unsqueeze(1).repeat(1, path_h.size(1), 1),
            rel_r_emb.unsqueeze(1).repeat(1, path_r.size(1), 1),
            ent_t_emb.unsqueeze(1).repeat(1, path_t.size(1), 1),
            path1_h_emb,
            path1_r_emb,
            path1_t_emb
        ], dim=-1)  # [batch, path_length, 600] 假设 ent_emd_dim=200 和 rel_emd_dim=200

        path1_gnn_output = self.path_gnn(path1_input)  # [batch, path_length, 100]
        path1_cnn_output = self.path_cnn(path1_gnn_output)  # [batch, new_seq_len, 100]
        path1_gru_output = self.path_gru(path1_cnn_output)  # [batch, new_seq_len, 100]
        path1_output = torch.mean(path1_gru_output, dim=1)  # 全局平均池化 [batch, 100]
        path1_value = self.fc_path1(path1_output)  # [batch, 1]

        # Path 2
        path2_h_emb = self.ent_embedding(path2_h)  # [batch, path_length, ent_emd_dim]
        path2_t_emb = self.ent_embedding(path2_t)
        path2_r_emb = self.rel_embedding(path2_r)

        path2_input = torch.cat([
            ent_h_emb.unsqueeze(1).repeat(1, path2_h.size(1), 1),
            rel_r_emb.unsqueeze(1).repeat(1, path2_r.size(1), 1),
            ent_t_emb.unsqueeze(1).repeat(1, path2_t.size(1), 1),
            path2_h_emb,
            path2_r_emb,
            path2_t_emb
        ], dim=-1)  # [batch, path_length, 600]

        path2_gnn_output = self.path_gnn(path2_input)  # [batch, path_length, 100]
        path2_cnn_output = self.path_cnn(path2_gnn_output)  # [batch, new_seq_len, 100]
        path2_gru_output = self.path_gru2(path2_cnn_output)  # [batch, new_seq_len, 100]
        path2_output = torch.mean(path2_gru_output, dim=1)  # 全局平均池化 [batch, 100]
        path2_value = self.fc_path2(path2_output)  # [batch, 1]

        # Path 3
        path3_h_emb = self.ent_embedding(path3_h)  # [batch, path_length, ent_emd_dim]
        path3_t_emb = self.ent_embedding(path3_t)
        path3_r_emb = self.rel_embedding(path3_r)

        path3_input = torch.cat([
            ent_h_emb.unsqueeze(1).repeat(1, path3_h.size(1), 1),
            rel_r_emb.unsqueeze(1).repeat(1, path3_r.size(1), 1),
            ent_t_emb.unsqueeze(1).repeat(1, path3_t.size(1), 1),
            path3_h_emb,
            path3_r_emb,
            path3_t_emb
        ], dim=-1)  # [batch, path_length, 600]

        path3_gnn_output = self.path_gnn(path3_input)  # [batch, path_length, 100]
        path3_cnn_output = self.path_cnn(path3_gnn_output)  # [batch, new_seq_len, 100]
        path3_gru_output = self.path_gru3(path3_cnn_output)  # [batch, new_seq_len, 100]
        path3_output = torch.mean(path3_gru_output, dim=1)  # 全局平均池化 [batch, 100]
        path3_value = self.fc_path3(path3_output)  # [batch, 1]

        # TransE 和 RRank
        rrank_value = self.fc_rrank(rrank)  # [batch, 1]

        # Combine features
        transE = transE.unsqueeze(-1)  # [batch, 1]
        combined_input = torch.cat([path1_value, path2_value, path3_value, transE, rrank_value], dim=-1)  # [batch, 5]

        # 最终分类层
        output = self.fc_bp(combined_input)  # [batch, 2]

        return output


def SelectModel(modelname, entvocabsize, relvocabsize, ent2vec, rel2vec, input_path_length, ent_emd_dim, rel_emd_dim):
    """
    根据模型名称选择并返回对应的模型实例。

    :param modelname: 模型名称
    :param entvocabsize: 实体词汇大小
    :param relvocabsize: 关系词汇大小
    :param ent2vec: 实体的预训练嵌入向量
    :param rel2vec: 关系的预训练嵌入向量
    :param input_path_length: 输入路径长度
    :param ent_emd_dim: 实体嵌入维度
    :param rel_emd_dim: 关系嵌入维度
    :return: 模型实例
    """

    if modelname == 'TRG-Trust':
        # 创建并返回 create_Model_BiLSTM_BP 类的实例
        return create_Model_BiLSTM_BP(
            entvocabsize=entvocabsize,
            relvocabsize=relvocabsize,
            ent2vec=ent2vec,
            rel2vec=rel2vec,
            input_path_length=input_path_length,
            ent_emd_dim=ent_emd_dim,
            rel_emd_dim=rel_emd_dim
        )

    # 如果模型名称不匹配，则抛出异常
    raise ValueError(f"Model name {modelname} not recognized.")

def train_model(modelname, datafile, modelfile, resultdir, npochos, batch_size, retrain=False):
    # 加载数据
    with open(datafile, 'rb') as f:
        (
            ent_vocab, ent_idex_word, rel_vocab, rel_idex_word,
            entity2vec, entity2vec_dim,
            relation2vec, relation2vec_dim,
            train_triple, train_confidence,
            test_triple, test_confidence,
            test_triple_KGC_h_t, test_confidence_KGC_h_t,
            test_triple_KGC_hr_, test_confidence_KGC_hr_,
            test_triple_KGC__rt, test_confidence_KGC__rt,
            tcthreshold_dict, train_transE, test_transE,
            test_transE_h_t, test_transE_hr_, test_transE__rt,
            rrkthreshold_dict, train_rrank, test_rrank,
            test_rrank_KGC_h_t, test_rrank_KGC_hr_, test_rrank_KGC__rt,
            max_p,
            train_path_h, train_path_t, train_path_r,
            test_path_h, test_path_t, test_path_r,
            train_path2_h, train_path2_t, train_path2_r,
            test_path2_h, test_path2_t, test_path2_r,
            train_path3_h, train_path3_t, train_path3_r,
            test_path3_h, test_path3_t, test_path3_r,
            test_path_h_KGC_h_t, test_path_t_KGC_h_t, test_path_r_KGC_h_t,
            test_path_h_hr_, test_path_t_hr_, test_path_r_hr_,
            test_path_h__rt, test_path_t__rt, test_path_r__rt
        ) = pickle.load(f)

    # 转换数据
    train_triple = np.array(train_triple)
    input_train_h = torch.tensor(train_triple[:, 0], dtype=torch.long)
    input_train_t = torch.tensor(train_triple[:, 1], dtype=torch.long)
    input_train_r = torch.tensor(train_triple[:, 2], dtype=torch.long)

    train_dataset = TensorDataset(
        input_train_h,
        input_train_t,
        input_train_r,
        torch.tensor(train_path_h, dtype=torch.long),
        torch.tensor(train_path_t, dtype=torch.long),
        torch.tensor(train_path_r, dtype=torch.long),
        torch.tensor(train_path2_h, dtype=torch.long),
        torch.tensor(train_path2_t, dtype=torch.long),
        torch.tensor(train_path2_r, dtype=torch.long),
        torch.tensor(train_path3_h, dtype=torch.long),
        torch.tensor(train_path3_t, dtype=torch.long),
        torch.tensor(train_path3_r, dtype=torch.long),
        torch.tensor(train_transE, dtype=torch.float32),
        torch.tensor(train_rrank, dtype=torch.float32),
        torch.tensor(train_confidence, dtype=torch.float32)
    )

    # 动态划分训练集和验证集
    dataset_size = len(train_dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    split = int(0.8 * dataset_size) # 20%作为验证集
    train_indices, val_indices = indices[:split], indices[split:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    # 初始化模型
    nn_model = SelectModel(
        modelname,
        entvocabsize=len(ent_vocab),
        relvocabsize=len(rel_vocab),
        ent2vec=entity2vec,
        rel2vec=relation2vec,
        input_path_length=max_p,
        ent_emd_dim=entity2vec_dim,
        rel_emd_dim=relation2vec_dim
    )

    if retrain:
        nn_model.load_state_dict(torch.load(modelfile))

    nn_model = nn_model.cuda() if torch.cuda.is_available() else nn_model

    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-5)  # L2正则化
    criterion = torch.nn.CrossEntropyLoss()

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6, verbose=True)

    maxF = 0
    early_stopping = 0

    # 开始训练
    for epoch in range(npochos):
        nn_model.train()
        total_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{npochos}", unit="batch") as pbar:
            for batch in pbar:
                inputs = [x.cuda() for x in batch[:-1]] if torch.cuda.is_available() else batch[:-1]
                labels = batch[-1].cuda() if torch.cuda.is_available() else batch[-1]
                labels = torch.argmax(labels, dim=1)

                optimizer.zero_grad()
                outputs = nn_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=5.0)

                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"Loss": f"{total_loss / (pbar.n + 1):.4f}"})

        print(f"Epoch {epoch + 1}/{npochos}, Loss: {total_loss:.4f}")

        # 验证模型
        nn_model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with tqdm(val_loader, desc="Validating", unit="batch") as pbar:
            with torch.no_grad():
                for batch in pbar:
                    inputs = [x.cuda() for x in batch[:-1]] if torch.cuda.is_available() else batch[:-1]
                    labels = batch[-1].cuda() if torch.cuda.is_available() else batch[-1]
                    labels = torch.argmax(labels, dim=1)

                    outputs = nn_model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    pbar.set_postfix({"Val Loss": f"{val_loss / (pbar.n + 1):.4f}"})

        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")

        # 学习率调度
        scheduler.step(acc)

        if acc > maxF:
            maxF = acc
            early_stopping = 0
            torch.save(nn_model.state_dict(), modelfile)
        else:
            early_stopping += 1

        print(f"Epoch {epoch + 1}, Max Accuracy: {maxF:.4f}, Early stopping counter: {early_stopping}")

        if early_stopping >= 12:
            print("Early stopping.")
            break

    return nn_model


# 目的是生成训练和验证数据的批次
def get_training_xy_otherset(train_triple, train_confidence, dev_triple, dev_confidence, train_transE, dev_transE,
                             train_rrank, dev_rrank, train_path_h, train_path_t, train_path_r, dev_path_h, dev_path_t,
                             dev_path_r, max_p, shuffle):
    assert len(train_triple) == len(
        train_confidence)  # 用于确保train_triple（训练三元组）和train_confidence（训练置信度）的长度相等。如果长度不相等，会触发AssertionError异常。
    indices = np.arange(len(train_confidence))  # 代码创建一个NumPy数组indices，其中包含了train_confidence的索引，用于在后续的数据处理中进行随机化操作。
    if shuffle:  # 检查shuffle参数的值
        np.random.shuffle(indices)  # 如果shuffle为True，则对indices数组中的索引进行随机洗牌操作，以便在训练数据中随机选择样本

    input_train_h = np.zeros((len(train_triple), 1)).astype('int32')  # 初始化了多个数组，用于存储训练数据的不同部分
    input_train_t = np.zeros((len(train_triple), 1)).astype('int32')
    input_train_r = np.zeros((len(train_triple), 1)).astype('int32')

    input_train_path_h = np.zeros((len(train_path_h), max_p)).astype('int32')
    input_train_path_t = np.zeros((len(train_path_h), max_p)).astype('int32')
    input_train_path_r = np.zeros((len(train_path_h), max_p)).astype('int32')
    y = np.zeros((len(train_confidence), 2)).astype('float32')
    input_train_transE = np.zeros((len(train_transE), 1)).astype('float32')
    input_train_rrank = np.zeros((len(train_rrank), 1)).astype('float32')

    for idx, s in enumerate(indices):  # 遍历indices中的随机化索引，用于填充训练数据的各个部分
        # print(s, len(train_path_h))
        # print(train_path_h[s])
        input_train_h[idx,] = train_triple[s][0]
        input_train_t[idx,] = train_triple[s][1]
        input_train_r[idx,] = train_triple[s][2]
        input_train_path_h[idx,] = train_path_h[s]
        input_train_path_t[idx,] = train_path_t[s]
        input_train_path_r[idx,] = train_path_r[s]
        y[idx,] = train_confidence[s]  # 置信度
        input_train_transE[idx,] = train_transE[s]  # 将train_transE中的第s个样本的值赋给input_train_transE数组中的第idx行。
        input_train_rrank[idx,] = train_rrank[s]
        # for idx2, word in enumerate(train_confidence[s]):
        #     targetvec = np.zeros(2).astype('int32')
        #     targetvec[word] = 1
        #     y[idx, ] = targetvec

    input_dev_h = np.zeros((len(dev_triple), 1)).astype(
        'int32')  # 数组用于存储验证集中的实体和关系 数组的长度为len(dev_triple)，这是验证集中三元组的数量 数组的形状是(len(dev_triple), 1)
    input_dev_t = np.zeros((len(dev_triple), 1)).astype('int32')
    input_dev_r = np.zeros((len(dev_triple), 1)).astype('int32')
    for idx, tri in enumerate(dev_triple):  # 用于遍历验证集中的三元组。dev_triple包含了验证集中的实体和关系信息。
        input_dev_h[idx,] = tri[0]
        input_dev_t[idx,] = tri[1]
        input_dev_r[idx,] = tri[2]

    input_dev_path_h = np.array(dev_path_h[
                                :])  # 创建了一个名为input_dev_path_h的数组，用于存储验证集中的路径的头部实体信息。dev_path_h包含了验证集中路径的头部实体信息。通过np.array(dev_path_h[:])将其复制到新的NumPy数组中。
    input_dev_path_t = np.array(dev_path_t[:])
    input_dev_path_r = np.array(dev_path_r[:])
    y_dev = np.array(dev_confidence[:])
    input_dev_transE = np.array(dev_transE[:])  # input_dev_transE和input_dev_rrank数组分别用于存储验证集中的TransE和RRank模型的相关信息
    input_dev_rrank = np.array(dev_rrank[:])

    yield input_train_h, input_train_t, input_train_r, \
          input_dev_h, input_dev_t, input_dev_r, \
          input_train_path_h, input_train_path_t, input_train_path_r, y, \
          input_dev_path_h, input_dev_path_t, input_dev_path_r, y_dev, \
          input_train_transE, input_dev_transE, \
          input_train_rrank, input_dev_rrank  # yield语句用于生成一个数据批次，并将多个数据数组以及它们的名称打包在一起。当生成器函数被调用时，返回给调用者





def save_model(nn_model, NN_MODEL_PATH):
    nn_model.save_weights(NN_MODEL_PATH, overwrite=True)  # 用于保存深度学习模型的权重参数到指定的文件路径NN_MODEL_PATH


# 目的是评估模型在测试数据上的性能，通过计算准确率来衡量模型的整体性能
def test_model_o(model,
               input_test_h, input_test_t, input_test_r,
               test_path_h, test_path_t, test_path_r,
               test_path2_h, test_path2_t, test_path2_r,
               test_path3_h, test_path3_t, test_path3_r,
               test_transE, test_rrank, test_confidence, resultfile):  # 函数的定义，接受多个参数

    total_predict_right = 0.  # 用于记录总共正确预测的数量，初始值为0.0。
    total_predict = 0.  # 用于记录总共进行的预测数量，初始值为0.0。
    total_right = 0.  # 用于记录总共正确的标签数量，初始值为0.0。
    results = model.predict([
        np.array(input_test_h), np.array(input_test_t), np.array(input_test_r),
        np.array(test_path_h), np.array(test_path_t), np.array(test_path_r),
        np.array(test_path2_h), np.array(test_path2_t), np.array(test_path2_r),
        np.array(test_path3_h), np.array(test_path3_t), np.array(test_path3_r),
        np.array(test_transE),
        np.array(test_rrank)
    ], batch_size=64)  # 使用深度学习模型(model)进行预测，将输入数据传递给模型，得到预测结果(results)。预测结果是一个包含两列的数组，每一列表示一个类别的概率。

    fin0 = open(resultfile + 'train_conf0.txt', 'w')  # 打开一个文件用于保存类别0的预测概率。
    fin1 = open(resultfile + 'train_conf1.txt', 'w')  # 打开一个文件用于保存类别1的预测概率。

    # result是一个一行二列的数组，是预测conf_train2id.txt中三元组的预测值。根据第一列表示预测为0（不正确）的预测值，第二列表示预测为1（正确）的预测值，预测值越接近1表示越真
    for i, res in enumerate(results):  # 遍历模型的预测结果，i表示索引，res表示每个样本的预测结果。

        #print("res:",test_confidence[i][1])
        # print(res)
        tag = np.argmax(res)  # 根据为0或1的预测值，找到预测概率最大的类别的索引，将其赋给tag。0或1

        # print(res)
        # tag = 0
        # if res >=0.5:
        #     tag = 1
        if test_confidence[i][1] == 1:  # test_confidence[i]是[0,1]或者[1,0]的数组，现在取第二位判断
            fin1.write(str(res[1]) + '\n')  # 如果test_confidence[i][1]是 1，则将res[1]值写入
            if tag == 1:
                total_predict_right += 1.0  # 如果tag等于1，表示与test_confidence[i][1]相等表示预测正确，将total_predict_right增加1。
        else:
            fin0.write(str(res[1]) + '\n')  # 将类别0的预测概率写入类别0的文件中
            if tag == 0:
                total_predict_right += 1.0

    fin0.close()  # 关闭保存类别0预测概率的文件
    fin1.close()
    print('total_predict_right', total_predict_right, 'len(test_confidence)', float(len(test_confidence)))
    acc = total_predict_right / float(len(test_confidence))  # 计算准确率，将正确的预测数量除以总样本数。
    return acc


#test测试生成的文件，可以作为散点图数据来源
def test_model(model,
              input_test_h, input_test_t, input_test_r,
              test_path_h, test_path_t, test_path_r,
              test_path2_h, test_path2_t, test_path2_r,
              test_path3_h, test_path3_t, test_path3_r,
              test_transE, test_rrank, test_confidence,  # 真实标签（独热编码）
              resultfile, batch_size=64):
    """
    使用 PyTorch 实现的模型测试函数。

    参数:
        model: 已训练好的 PyTorch 模型。
        input_test_h, input_test_t, input_test_r: 第一组测试输入。
        test_path_h, test_path_t, test_path_r: 第二组测试输入。
        test_path2_h, test_path2_t, test_path2_r: 第三组测试输入。
        test_path3_h, test_path3_t, test_path3_r: 第四组测试输入。
        test_transE, test_rrank: 其他测试输入。
        test_confidence: 样本的真实标签（独热编码，如 [0, 1] 或 [1, 0]）。
        resultfile: 结果文件存储的目录。
        batch_size: 每个批次的大小。

    返回:
        acc: 准确率。
        recall: 召回率。
    """
    # 打印标签的形状以确认格式
    print("Shape of test_confidence:", np.array(test_confidence).shape)

    # 设置模型为评估模式
    model.eval()

    # 确定设备（GPU 或 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 将 NumPy 数组转换为 PyTorch 张量并移动到设备
    input_test_h = torch.tensor(input_test_h, dtype=torch.long)
    input_test_t = torch.tensor(input_test_t, dtype=torch.long)
    input_test_r = torch.tensor(input_test_r, dtype=torch.long)
    test_path_h = torch.tensor(test_path_h, dtype=torch.long)
    test_path_t = torch.tensor(test_path_t, dtype=torch.long)
    test_path_r = torch.tensor(test_path_r, dtype=torch.long)
    test_path2_h = torch.tensor(test_path2_h, dtype=torch.long)
    test_path2_t = torch.tensor(test_path2_t, dtype=torch.long)
    test_path2_r = torch.tensor(test_path2_r, dtype=torch.long)
    test_path3_h = torch.tensor(test_path3_h, dtype=torch.long)
    test_path3_t = torch.tensor(test_path3_t, dtype=torch.long)
    test_path3_r = torch.tensor(test_path3_r, dtype=torch.long)
    test_transE = torch.tensor(test_transE, dtype=torch.float32)
    test_rrank = torch.tensor(test_rrank, dtype=torch.float32)

    # 处理标签
    test_confidence = np.array(test_confidence)
    if test_confidence.ndim > 1:
        # 假设标签是独热编码，转换为类别索引
        test_confidence = np.argmax(test_confidence, axis=1)
        print("Converted one-hot labels to class indices.")
    else:
        # 假设标签已经是一维数组
        test_confidence = test_confidence.astype(int)
        print("Labels are single class indices.")

    test_confidence = torch.tensor(test_confidence, dtype=torch.long)

    # 将所有张量移动到设备
    input_test_h = input_test_h.to(device)
    input_test_t = input_test_t.to(device)
    input_test_r = input_test_r.to(device)
    test_path_h = test_path_h.to(device)
    test_path_t = test_path_t.to(device)
    test_path_r = test_path_r.to(device)
    test_path2_h = test_path2_h.to(device)
    test_path2_t = test_path2_t.to(device)
    test_path2_r = test_path2_r.to(device)
    test_path3_h = test_path3_h.to(device)
    test_path3_t = test_path3_t.to(device)
    test_path3_r = test_path3_r.to(device)
    test_transE = test_transE.to(device)
    test_rrank = test_rrank.to(device)
    test_confidence = test_confidence.to(device)

    # 创建 TensorDataset 和 DataLoader
    dataset = TensorDataset(input_test_h, input_test_t, input_test_r,
                            test_path_h, test_path_t, test_path_r,
                            test_path2_h, test_path2_t, test_path2_r,
                            test_path3_h, test_path3_t, test_path3_r,
                            test_transE, test_rrank, test_confidence)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 初始化统计变量
    total_predict_right = 0.0  # 正确预测的数量
    true_positives = 0  # TP: 正确预测为1的数量
    false_negatives = 0  # FN: 实际为1但预测为0的数量
    total_samples = len(test_confidence)  # 总样本数量

    # 确保结果目录存在
    os.makedirs(resultfile, exist_ok=True)
    #散点图来源
    resultfile_conf0 = os.path.join(resultfile, 'train_conf0.txt')
    resultfile_conf1 = os.path.join(resultfile, 'train_conf1.txt')

    # 初始化缓冲区
    conf0_buffer = []
    conf1_buffer = []

    # 打开文件用于写入预测置信度
    try:
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Testing"):
                # 分离输入和标签
                inputs = [batch[i].to(device) for i in range(14)]  # 前14个元素为输入
                labels_batch = batch[14].to(device)  # 第15个元素为标签（单一类别）

                # 前向传播
                outputs = model(inputs)  # 输出形状假设为 (batch_size, num_classes)

                # 计算概率（假设为二分类问题，使用 softmax）
                probabilities = torch.softmax(outputs, dim=1)  # 转换为概率

                # 获取类别1的概率作为置信度
                confidences = probabilities[:, 1]  # 类别1的概率

                # 获取预测标签
                predictions = torch.argmax(probabilities, dim=1)  # 预测的类别标签

                # 将张量转换为 CPU 上的 NumPy 数组
                actual_labels = labels_batch.cpu().numpy()  # (batch_size,)
                confidences_np = confidences.cpu().numpy()
                predictions_np = predictions.cpu().numpy()

                # 遍历批次中的每个样本
                for i in range(len(predictions_np)):
                    tag = predictions_np[i]  # 预测标签（0 或 1）
                    confidence = confidences_np[i]  # 类别1的置信度
                    actual_label = actual_labels[i]  # 实际标签（0 或 1）

                    if actual_label == 1:  # 实际标签为1
                        conf1_buffer.append(f"{confidence}\n")
                        if tag == 1:
                            total_predict_right += 1.0
                            true_positives += 1
                        else:
                            false_negatives += 1
                    else:  # 实际标签为0
                        conf0_buffer.append(f"{confidence}\n")
                        if tag == 0:
                            total_predict_right += 1.0

        # 批量写入文件
        with open(resultfile_conf1, 'w') as fin1:
            fin1.writelines(conf1_buffer)
        with open(resultfile_conf0, 'w') as fin0:
            fin0.writelines(conf0_buffer)

    except Exception as e:
        print(f"An error occurred during testing: {e}")
        return 0.0, 0.0

    # 计算准确率
    acc = total_predict_right / float(total_samples) if total_samples > 0 else 0.0

    # 计算召回率
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    print('Total Predict Right:', total_predict_right)
    print('Total Samples:', total_samples)
    print('Accuracy:', acc)
    print('Recall:', recall)

    return acc, recall







# 用于计算分类模型性能指标的函数，通常用于二分类问题。在这个函数中，主要计算 Precision（精确率）、Recall（召回率）和 F1 Score（F1 分数）等性能指标，并在不同的阈值下绘制 Precision-Recall 曲线。
def test_model_PR(model,
                  input_test_h, input_test_t, input_test_r,
                  test_path_h, test_path_t, test_path_r,
                  test_path2_h, test_path2_t, test_path2_r,
                  test_path3_h, test_path3_t, test_path3_r,
                  test_transE, test_rrank, test_confidence, resultfile):  # 传递参数

    results = model.predict([
        np.array(input_test_h), np.array(input_test_t), np.array(input_test_r),
        np.array(test_path_h), np.array(test_path_t), np.array(test_path_r),
        np.array(test_path2_h), np.array(test_path2_t), np.array(test_path2_r),
        np.array(test_path3_h), np.array(test_path3_t), np.array(test_path3_r),
        np.array(test_transE),
        np.array(test_rrank)
    ], batch_size=64)  # 使用深度学习模型(model)进行预测，将输入数据传递给模型，得到预测结果(results)。预测结果是一个包含两列的数组，每一列表示一个类别的概率。

    fw = open(resultfile + 'RP.txt', 'w')
    total_predict_right = 0.
    Plist0 = []
    Rlist0 = []
    PRlist = []
    maxF = 0.0  # 精确率和召回率的调和平均数的最大值，最大为1，最小为0。
    for i, tri in enumerate(test_confidence):
        PRlist.append((results[i][1], test_confidence[i][1]))
    PRlist = sorted(PRlist, key=lambda sp: sp[0], reverse=True)

    for i, res in enumerate(PRlist):
        if res[1] == 1:
            total_predict_right += 1.0
        # if (i + 1) % (3470) == 0:   #值应该是总测试样本数除以 batch size 的结果，表示每个 epoch 中迭代的次数
        if (i + 1) % (3470) == 0:
            P = total_predict_right / (i + 1)
            R = total_predict_right / (len(PRlist) * 0.5)
            #(i, total_predict_right, P, R)
            if P + R ==0:
                P =0.000001
            F = 2 * P * R / (P + R)
            if maxF < F:
                maxF = F
            Rlist0.append(R)
            Plist0.append(P)
            #print(total_predict_right / (i + 1), '\t', total_predict_right / (len(PRlist) * 0.5), 'maxF = ', maxF)
            fw.write(f"Recall= {str(R)}\tPrecision= {str(P)}\tmaxF = {maxF}\n")
    fw.close()

    Thresholdlist = []  # 创建一个空列表，用于存储不同阈值的 Precision 和 Recall 数据点
    Plist1 = []
    Rlist1 = []
    Plist0 = []
    Rlist0 = []
    maxF = 0.0  # 记录最大的 F1 分数
    th = 0.01  # 起始阈值
    # fw = open(resultfile + 'RP.txt', 'w')
    while th <= 1.0:
        total_predict_right = 0.0  # 用于记录正确预测的数量，初始值为0.0
        total_predict_right0 = 0.0  # 用于记录正确预测的数量，初始值为0.0
        total_predict = 0.00001  # 用于记录总共进行的预测数量
        total_right = 0.00001  # 用于记录总共正确的标签数量
        for i, res in enumerate(results):
            tag = 0
            if res[1] >= th:  # 如果类别1的预测概率大于等于当前阈值 th，将 tag 设置为1，表示类别1。
                tag = 1
                total_predict += 1.0

            if test_confidence[i][1] == 1:
                if tag == 1:
                    total_predict_right += 1.0
                total_right += 1.0
            else:
                if tag == 0:
                    total_predict_right0 += 1.0

        P0 = total_predict_right0 / (len(results) - total_predict)  # 计算类别0的 Precision，即类别0的正确预测数量除以所有预测为类别0的数量
        R0 = total_predict_right0 / (len(results) - total_right)  # 计算类别0的 Recall，即类别0的正确预测数量除以所有真实为类别0的样本数量
        Plist0.append(P0)  # 将类别0的 Precision 和 Recall 分别添加到 Plist0 和 Rlist0 列表中。
        Rlist0.append(R0)
        P = total_predict_right / total_predict  # 计算类别1的 Precision，即类别1的正确预测数量除以所有预测为类别1的数量。
        R = total_predict_right / total_right  # 计算类别1的 Recall，即类别1的正确预测数量除以所有真实为类别1的样本数量。
        F = 2 * P * R / (P + R + 0.00001)  # 计算 F1 分数，使用 Precision 和 Recall 计算。这是二元分类问题中常用的综合评价指标，用于衡量分类器的性能。
        if maxF < F:
            maxF = F  # 更新 maxF 为当前 F1 分数
        #print('threshold = ', th, R, P, 'maxF = ', maxF)
        Thresholdlist.append(th)  # 将当前阈值、类别1的 Recall 和 Precision 添加到相应的列表中，用于后续分析或可视化。
        Rlist1.append(R)
        Plist1.append(P)
        # fw.write(str(R) + '\t' + str(P) + '\n')

        th += 0.02  # 增加阈值 th 的值，以继续下一轮循环，直到 th 大于1.0，完成整个阈值搜索的过程。

    # # fw.close()
    # # 使用 Matplotlib 库创建一个图形来展示 Precision 和 Recall 随阈值的变化情况
    # a = plt.subplot(1, 1, 1)
    #
    # # 这里b表示blue，g表示green，r表示red，-表示连接线，--表示虚线链接
    # a1 = a.plot(Thresholdlist, Plist1, 'b-', label='Precision')
    # a2 = a.plot(Thresholdlist, Rlist1, 'r-', label='Reall')
    # # a3 = a.plot(Thresholdlist, Plist0, 'b--', label='P_negitive')
    # # a4 = a.plot(Thresholdlist, Rlist0, 'r--', label='R_negitive')
    # # a3 = a.plot(Thresholdlist, Flist, 'y-', label='F')
    # # a4 = a.plot(Rlist, Plist, 'ro-', label='R-P')
    #
    # # 标记图的题目，x和y轴
    # # plt.title("The Precision, Recall, F-values during different triple confidence threshold")
    # plt.xlabel("Triple Trustworthiness")
    # plt.ylabel("Values")
    #
    # # 显示图例
    # handles, labels = a.get_legend_handles_labels()
    # a.legend(handles[::-1], labels[::-1])
    # plt.show()
    # # #这
    # b = plt.subplot(1, 1, 1)
    # # 这里b表示blue，g表示green，r表示red，-表示连接线，--表示虚线链接
    # # b1 = b.plot(Rlist, Plist, 'bx-', label='R-P')
    # b2 = b.plot(Rlist0, Plist0, 'go-', label='R-F')
    # # 标记图的题目，x和y轴
    # # plt.title("The Precision, Recall, F-values during different triple confidence threshold")
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # # 显示图例
    # handles, labels = b.get_legend_handles_labels()
    # b.legend(handles[::-1], labels[::-1])
    # plt.show()


# 该函数评估链路预测模型在测试数据上的性能。它使用 TransE 和 RRank 模型来计算损坏的三元组（其中一个实体被替换）的置信度分数。机器学习模型用于预测损坏的三元组中正确实体的可能性。它计算各种指标，例如原始排名和过滤排名、头实体和尾实体的命中数为 10 以及命中数为 1
def test_model_linkPrediction(model, datafile, entityRank):
    '''#ent_vocab, rel_vocab: 实体和关系的词汇表。entity2vec, entity2vec_dim, relation2vec, relation2vec_dim: 实体和关系的嵌入向量及其维度。train_triple, train_confidence: 训练数据的三元组和置信度。
    #test_triple, test_confidence: 测试数据的三元组和置信度。tcthreshold_dict: 存储了TransE模型的置信度阈值。train_transE, test_transE: 训练和测试数据的TransE模型置信度。rrkthreshold_dict: 存储了RRank模型的置信度阈值。
    #train_rrank, test_rrank: 训练和测试数据的RRank模型置信度。max_p: 最大路径长度。train_path_h, train_path_t, train_path_r: 训练数据的路径头、路径尾、路径关系。test_path_h, test_path_t, test_path_r: 测试数据的路径头、路径尾、路径关系。
    ent_vocab, rel_vocab, \
    entity2vec, entity2vec_dim, \
    relation2vec, relation2vec_dim, \
    train_triple, train_confidence, \
    test_triple, test_confidence, \
    tcthreshold_dict, train_transE, test_transE,\
    rrkthreshold_dict, train_rrank, test_rrank,\
    max_p, \
    train_path_h, train_path_t, train_path_r,\
    test_path_h, test_path_t, test_path_r = pickle.load(open(datafile, 'rb'))'''
    ent_vocab, ent_idex_word, rel_vocab, rel_idex_word, \
    entity2vec, entity2vec_dim, \
    relation2vec, relation2vec_dim, \
    train_triple, train_confidence, \
    test_triple, test_confidence, \
    test_triple_KGC_h_t, test_confidence_KGC_h_t, \
    test_triple_KGC_hr_, test_confidence_KGC_hr_, \
    test_triple_KGC__rt, test_confidence_KGC__rt, \
    tcthreshold_dict, train_transE, test_transE, \
    test_transE_h_t, \
    test_transE_hr_, \
    test_transE__rt, \
    rrkthreshold_dict, train_rrank, test_rrank, \
    test_rrank_KGC_h_t, \
    test_rrank_KGC_hr_, \
    test_rrank_KGC__rt, \
    max_p, \
    train_path_h, train_path_t, train_path_r, \
    test_path_h, test_path_t, test_path_r, \
    train_path2_h, train_path2_t, train_path2_r, \
    test_path2_h, test_path2_t, test_path2_r, \
    train_path3_h, train_path3_t, train_path3_r, \
    test_path3_h, test_path3_t, test_path3_r, \
    test_path_h_KGC_h_t, test_path_t_KGC_h_t, test_path_r_KGC_h_t, \
    test_path_h_KGC_hr_, test_path_t_KGC_hr_, test_path_r_KGC_hr_, \
    test_path_h_KGC__rt, test_path_t_KGC__rt, test_path_r_KGC__rt = pickle.load(open(datafile, 'rb'))

    dict_entityRank = get_dict_entityRank(entityRank)  # 存储了实体的排名字典，用于计算排名
    goldtriples = get_goldtriples()  # 存储了正确的三元组数据，用于过滤计算过滤排名的正确结果

    totalRawHeadRank = 0.  # 初始化了用于存储原始排名的头尾实体
    totalRawTailRank = 0.
    totalFilterHeadRank = 0.  # 初始化了用于过滤排名的头尾实体
    totalFilterTailRank = 0.

    totalRawHeadHit10 = 0.
    totalRawTailHit10 = 0.
    totalRawHeadHit1 = 0.
    totalRawTailHit1 = 0.

    totalFilterHeadHit10 = 0.
    totalFilterTailHit10 = 0.
    totalFilterHeadHit1 = 0.
    totalFilterTailHit1 = 0.

    rawTailList = []
    rawHeadList = []
    filterTailList = []
    filterHeadList = []
    for i in range(len(test_triple)):

        rawTailList.clear()
        filterTailList.clear()
        changetriples = []
        for corruptedTailEntity in ent_vocab.values():
            changetriples.append((test_triple[i][0], corruptedTailEntity, test_triple[i][2],
                                  1))  # 通过替换当前测试三元组中的尾部实体，保留头实体和关系来生成新的三元组changetriples列表。
        # tcthreshold_dict：存储TransE模型置信度阈值的字典。 changetriples：包含损坏的三元组的列表。这些损坏的三元组是通过用其他实体替换测试三元组中的原始尾部实体而创建的，同时保持头实体和关系相同entity2vec：存储实体嵌入的字典。relation2vec：存储关系嵌入的字典
        transE = get_TransConfidence_train(tcthreshold_dict, changetriples, entity2vec,
                                           relation2vec)  # 使用 TransE 模型计算损坏三元组置信度得分的函数  使用 TransE 模型计算每个损坏的三元组的置信度得分并将其存储在transE变量中
        rrank = get_RRankConfidence(rrkthreshold_dict, changetriples,
                                    dict_entityRank)  # 使用 RRank 模型计算损坏三元组置信度得分的函数  使用 RRank 模型计算每个损坏的三元组的置信度得分并将其存储在rrank变量中。

        results = model.predict([np.array(transE), np.array(
            rrank)])  # model用于链接预测的机器学习模型 np.array计算出的 TransE 和 RRank 置信度分数将转换为 NumPy 数组，以输入模型进行预测。model.predict：此方法用于根据计算的置信度分数来预测损坏三元组的尾部实体

        for r in range(len(results)):  # 此循环迭代 中每个损坏的三元组的链接预测模型的结果changetriples

            rawTailList.append((changetriples[r][1], results[r][
                1]))  # 对于每个损坏的三元组，它会将一个元组附加到rawTailList. 该元组包含：changetriples[r][1]：来自损坏的三元组的尾部实体。results[r][1]：模型对这个损坏的三元组的预测。
            if (changetriples[r][0], changetriples[r][1], changetriples[r][2],
                1) not in goldtriples:  # 检查数据集中是否存在原始三元组（损坏之前) 如果三元组不符合正确数据集标准
                filterTailList.append((changetriples[r][1], results[r][
                    1]))  # changetriples[r][1]：来自损坏的三元组的尾部实体。results[r][1]：模型对这个损坏的三元组的预测。

        rawTailList = sorted(rawTailList, key=lambda sp: sp[1],
                             reverse=True)  # 在处理所有损坏的三元组并附加其预测后，该行rawTailList根据置信度分数按降序排序 ( reverse=True)。这将为每个损坏的尾部实体创建一个预测排名列表。
        filterTailList = sorted(filterTailList, key=lambda sp: sp[1], reverse=True)  # filterTailList根据置信度分数按降序排序
        for j, tri in enumerate(rawTailList):  # 循环迭代原始尾部实体的排名预测，这些实体存储在rawTailList。
            j = j + 1  # 在每个循环内，j每次迭代都会加 1。该变量表示当前预测的排名。
            if tri[0] == test_triple[1]:  # 对于排名列表中的每个预测，它检查尾部实体 ( tri[0]) 是否与测试三元组 ( test_triple[1]) 中的实际尾部实体匹配。
                totalRawTailRank += j  # totalRawTailRank += j：将排名j添加到totalRawTailRank，累积原始设置中正确预测的尾部实体的排名。
                if j <= 10:
                    totalRawTailHit10 += 1.0
                if j == 1:
                    totalRawTailHit1 += 1.0
                break
        for j, tri in enumerate(filterTailList):  # 循环迭代过滤后尾部实体的排名预测，这些实体存储在filterTailList
            j = j + 1
            if tri[0] == test_triple[1]:
                totalFilterTailRank += j
                if j <= 10:
                    totalFilterTailHit10 += 1.0
                if j == 1:
                    totalFilterTailHit1 += 1.0
                break

    for i in range(len(test_triple)):

        rawHeadList.clear()
        filterHeadList.clear()
        changetriples = []
        for corruptedHeadEntity in ent_vocab.values():  # 相比于上一段代码  一个部分侧重于预测尾部实体，另一个部分侧重于预测测试三元组的头部实体。
            changetriples.append((corruptedHeadEntity, test_triple[i][1], test_triple[i][2],
                                  1))  # 代码迭代中的所有实体，ent_vocab并通过替换原始头实体同时保持尾实体和关系不变来生成损坏的三元组。

        transE = get_TransConfidence_train(tcthreshold_dict, changetriples, entity2vec, relation2vec)

        rrank = get_RRankConfidence(rrkthreshold_dict, changetriples, dict_entityRank)


        results = model.predict([np.array(transE), np.array(rrank)])

        for r in range(len(results)):
            rawHeadList.append((changetriples[r][1], results[r][1]))
            if (changetriples[r][0], changetriples[r][1], changetriples[r][2], 1) not in goldtriples:
                filterHeadList.append((changetriples[r][1], results[r][1]))

        rawHeadList = sorted(rawHeadList, key=lambda sp: sp[1], reverse=True)
        filterHeadList = sorted(filterHeadList, key=lambda sp: sp[1], reverse=True)
        for j, tri in enumerate(rawHeadList):
            j = j + 1
            if tri[0] == test_triple[1]:
                totalRawHeadRank += j
                if j <= 10:
                    totalRawHeadHit10 += 1.0
                if j == 1:
                    totalRawHeadHit1 += 1.0
                break
        for j, tri in enumerate(filterHeadList):
            j = j + 1
            if tri[0] == test_triple[1]:
                totalFilterHeadRank += j
                if j <= 10:
                    totalFilterHeadHit10 += 1.0
                if j == 1:
                    totalFilterHeadHit1 += 1.0
                break

    print("RAW_RANK: ", (totalRawTailRank + totalRawHeadRank) / float(2. * len(test_triple)))
    print("FILTER_RANK: ", (totalFilterHeadRank + totalFilterTailRank) / float(2. * len(test_triple)))
    print("RAW_HIT@10: ", (totalRawTailHit10 + totalRawHeadHit10) / float(2. * len(test_triple)))
    print("FILTER_HIT@10: ", (totalFilterHeadHit10 + totalFilterTailHit10) / float(2. * len(test_triple)))
    print("RAW_HIT@1: ", (totalRawHeadHit1 + totalRawTailHit1) / float(2. * len(test_triple)))
    print("FILTER_HIT@1: ", (totalFilterHeadHit1 + totalFilterTailHit1) / float(2. * len(test_triple)))


# 目的是在测试集上进行模型预测，并保存每个测试样本的预测结果和置信度得分。计算平均置信度有助于了解整体的置信度水平。
def test_model_load_old(model, ent_idex_word, rel_idex_word, test_triple,
                    input_test_h, input_test_t, input_test_r,
                    test_path_h, test_path_t, test_path_r,
                    test_path2_h, test_path2_t, test_path2_r,
                    test_path3_h, test_path3_t, test_path3_r,
                    test_transE, test_rrank, test_confidence, resultfile):
    # results = model.predict([
    #     np.array(test_path_h), np.array(test_path_t), np.array(test_path_r),
    #                         np.array(test_transE),
    #                         np.array(test_rrank)
    # ], batch_size=40)
    results = model.predict([
        np.array(input_test_h), np.array(input_test_t), np.array(input_test_r),
        np.array(test_path_h), np.array(test_path_t), np.array(test_path_r),
        np.array(test_path2_h), np.array(test_path2_t), np.array(test_path2_r),
        np.array(test_path3_h), np.array(test_path3_t), np.array(test_path3_r),
        np.array(test_transE),
        np.array(test_rrank)
    ], batch_size=64)

    # print(results)
    All_conf = 0.0  # 用于累积所有预测的置信度得分总和
    fr = open(resultfile + 'end.txt', 'w')  # 打开resultfile写入模式，创建或覆盖文件以写入结果

    for i, res in enumerate(results):
        conf = res[1]  # 行从模型的预测结果中提取置信度得分
        # print(conf)
        All_conf += conf
        strs = ent_idex_word[test_triple[i][0]] + '\t' + rel_idex_word[test_triple[i][2]] + \
               '\t' + ent_idex_word[test_triple[i][1]] + '\t' + str(conf) + '\n'
        fr.write(strs)
    fr.close()

    avg_conf = All_conf / float(len(results))  # 计算结果为平均置信度分数
    print('avg_conf is ... ', avg_conf)



#求平均置信度（链路预测时可以求解平置信度）
def test_model_load(model, ent_idex_word, rel_idex_word, test_triple,
                           input_test_h, input_test_t, input_test_r,
                           test_path_h, test_path_t, test_path_r,
                           test_path2_h, test_path2_t, test_path2_r,
                           test_path3_h, test_path3_t, test_path3_r,
                           test_transE, test_rrank, test_confidence, resultfile,
                           batch_size=64, device='cuda'):
    """
    使用PyTorch实现的模型测试函数，保持与原始test_model_load相同的计算方式。

    参数:
        model: 已训练好的PyTorch模型。
        ent_idex_word: 实体索引到词的映射字典。
        rel_idex_word: 关系索引到词的映射字典。
        test_triple: 测试三元组列表，每个三元组为(h, t, r)。
        input_test_h, input_test_t, input_test_r: 测试头实体、尾实体、关系的输入。
        test_path_h, test_path_t, test_path_r: 路径1的头实体、尾实体、关系。
        test_path2_h, test_path2_t, test_path2_r: 路径2的头实体、尾实体、关系。
        test_path3_h, test_path3_t, test_path3_r: 路径3的头实体、尾实体、关系。
        test_transE: TransE相关的测试数据。
        test_rrank: rrank相关的测试数据。
        test_confidence: 测试置信度（未在函数中使用，保留以保持参数一致）。
        resultfile: 结果文件的路径前缀。
        batch_size: 每个批次的大小。
        device: 使用的设备（'cpu'或'cuda'）。如果为None，将自动选择。
    """

    # 1. 设备选择
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 2. 数据转换
    input_test_h = torch.tensor(input_test_h, device=device, dtype=torch.long)
    input_test_t = torch.tensor(input_test_t, device=device, dtype=torch.long)
    input_test_r = torch.tensor(input_test_r, device=device, dtype=torch.long)
    test_path_h = torch.tensor(test_path_h, device=device, dtype=torch.long)
    test_path_t = torch.tensor(test_path_t, device=device, dtype=torch.long)
    test_path_r = torch.tensor(test_path_r, device=device, dtype=torch.long)
    test_path2_h = torch.tensor(test_path2_h, device=device, dtype=torch.long)
    test_path2_t = torch.tensor(test_path2_t, device=device, dtype=torch.long)
    test_path2_r = torch.tensor(test_path2_r, device=device, dtype=torch.long)
    test_path3_h = torch.tensor(test_path3_h, device=device, dtype=torch.long)
    test_path3_t = torch.tensor(test_path3_t, device=device, dtype=torch.long)
    test_path3_r = torch.tensor(test_path3_r, device=device, dtype=torch.long)
    test_transE = torch.tensor(test_transE, device=device, dtype=torch.float32)
    test_rrank = torch.tensor(test_rrank, device=device, dtype=torch.float32)

    # 3. 创建数据集和数据加载器
    dataset = TensorDataset(input_test_h, input_test_t, input_test_r,
                            test_path_h, test_path_t, test_path_r,
                            test_path2_h, test_path2_t, test_path2_r,
                            test_path3_h, test_path3_t, test_path3_r,
                            test_transE, test_rrank)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    All_conf = 0.0  # 累积所有预测的置信度得分总和
    total_samples = len(test_triple)

    # 4. 确保结果目录存在
    resultfile_path = os.path.join(resultfile, 'end.txt')
    os.makedirs(os.path.dirname(resultfile_path), exist_ok=True)

    # 5. 打开结果文件写入模式
    with open(resultfile_path, 'w', encoding='utf-8') as fr:
        with torch.no_grad():  # 禁用梯度计算
            for batch_idx, batch in enumerate(data_loader):
                # 将批次数据移动到设备
                batch = [x.to(device) for x in batch]

                # 模型预测
                outputs = model(batch)  # 假设模型输出形状为 (batch_size, 2)

                # 应用 Softmax 激活函数以获得每个类别的概率
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()  # 形状 (batch_size, 2)

                # 提取第二个类别的概率作为置信度
                confidences = probabilities[:, 1]  # 形状 (batch_size,)

                # 计算当前批次的起始索引
                start_idx = batch_idx * batch_size

                # 遍历当前批次的所有样本
                for i in range(len(confidences)):
                    idx = start_idx + i
                    if idx >= total_samples:
                        break  # 防止索引超出范围

                    triple = test_triple[idx]
                    if len(triple) < 3:
                        raise ValueError(f"test_triple[{idx}] has less than 3 elements: {triple}")
                    h_idx, t_idx, r_idx = triple[:3]  # 只取前三个元素

                    conf = confidences[i]

                    # 调试信息
                    if not (0.0 <= conf <= 1.0):
                        print(f"Warning: confidence {conf} for test_triple[{idx}] is out of bounds.")

                    All_conf += conf

                    # 构建输出字符串
                    output_str = f"{ent_idex_word[h_idx]}\t{rel_idex_word[r_idx]}\t{ent_idex_word[t_idx]}\t{conf}\n"
                    fr.write(output_str)

    # 6. 计算平均置信度
    avg_conf = All_conf / float(total_samples) if total_samples > 0 else 0.0
    print('平均置信度：', avg_conf)

    return avg_conf


'''def infer_model(modelname, entityRank, datafile, modelfile, resultfile, batch_size=50):
    ent_vocab, ent_idex_word, rel_vocab, rel_idex_word, \
    entity2vec, entity2vec_dim, \
    relation2vec, relation2vec_dim, \
    train_triple, train_confidence, \
    test_triple, test_confidence, \
    test_triple_KGC_h_t, test_confidence_KGC_h_t, \
    test_triple_KGC_hr_, test_confidence_KGC_hr_, \
    test_triple_KGC__rt, test_confidence_KGC__rt, \
    tcthreshold_dict, train_transE, test_transE, \
    test_transE_h_t, \
    test_transE_hr_, \
    test_transE__rt, \
    rrkthreshold_dict, train_rrank, test_rrank, \
    test_rrank_KGC_h_t, \
    test_rrank_KGC_hr_, \
    test_rrank_KGC__rt, \
    max_p, \
    train_path_h, train_path_t, train_path_r, \
    test_path_h, test_path_t, test_path_r, \
    train_path2_h, train_path2_t, train_path2_r, \
    test_path2_h, test_path2_t, test_path2_r, \
    train_path3_h, train_path3_t, train_path3_r, \
    test_path3_h, test_path3_t, test_path3_r, \
    test_path_h_KGC_h_t, test_path_t_KGC_h_t, test_path_r_KGC_h_t, \
    test_path_h_KGC_hr_, test_path_t_KGC_hr_, test_path_r_KGC_hr_, \
    test_path_h_KGC__rt, test_path_t_KGC__rt, test_path_r_KGC__rt = pickle.load(open(datafile, 'rb'))'''
def infer_model(modelname, entityRank, datafile, datafile_test,modelfile, resultfile, batch_size=50):
    ent_vocab, ent_idex_word, rel_vocab, rel_idex_word, \
    entity2vec, entity2vec_dim, \
    relation2vec, relation2vec_dim, \
    train_triple, train_confidence, \
    test_triple, test_confidence, \
    test_triple_KGC_h_t, test_confidence_KGC_h_t, \
    test_triple_KGC_hr_, test_confidence_KGC_hr_, \
    test_triple_KGC__rt, test_confidence_KGC__rt, \
    tcthreshold_dict, train_transE, test_transE, \
    test_transE_h_t, \
    test_transE_hr_, \
    test_transE__rt, \
    rrkthreshold_dict, train_rrank, test_rrank, \
    test_rrank_KGC_h_t, \
    test_rrank_KGC_hr_, \
    test_rrank_KGC__rt, \
    max_p, \
    train_path_h, train_path_t, train_path_r, \
    test_path_h, test_path_t, test_path_r, \
    train_path2_h, train_path2_t, train_path2_r, \
    test_path2_h, test_path2_t, test_path2_r, \
    train_path3_h, train_path3_t, train_path3_r, \
    test_path3_h, test_path3_t, test_path3_r, \
    test_path_h_KGC_h_t, test_path_t_KGC_h_t, test_path_r_KGC_h_t, \
    test_path_h_KGC_hr_, test_path_t_KGC_hr_, test_path_r_KGC_hr_, \
    test_path_h_KGC__rt, test_path_t_KGC__rt, test_path_r_KGC__rt = pickle.load(open(datafile, 'rb'))

    test_triple, test_confidence, \
    test_triple_KGC_h_t, test_confidence_KGC_h_t, \
    test_triple_KGC_hr_, test_confidence_KGC_hr_, \
    test_triple_KGC__rt, test_confidence_KGC__rt, \
    test_transE, \
    test_transE_h_t, \
    test_transE_hr_, \
    test_transE__rt, \
    test_rrank, \
    test_rrank_KGC_h_t, \
    test_rrank_KGC_hr_, \
    test_rrank_KGC__rt, \
    test_path_h, test_path_t, test_path_r, \
    test_path2_h, test_path2_t, test_path2_r, \
    test_path3_h, test_path3_t, test_path3_r, \
    test_path_h_KGC_h_t, test_path_t_KGC_h_t, test_path_r_KGC_h_t, \
    test_path_h_KGC_hr_, test_path_t_KGC_hr_, test_path_r_KGC_hr_, \
    test_path_h_KGC__rt, test_path_t_KGC__rt, test_path_r_KGC__rt = pickle.load(open(datafile_test, 'rb'))

    input_train_h = np.zeros((len(train_triple), 1)).astype('int32')  # 为训练数据创建的。这些矩阵存储每个训练三元组的头实体、尾实体和关系。
    input_train_t = np.zeros((len(train_triple), 1)).astype('int32')
    input_train_r = np.zeros((len(train_triple), 1)).astype('int32')
    for idx, s in enumerate(train_triple):
        input_train_h[idx,] = train_triple[idx][0]
        input_train_t[idx,] = train_triple[idx][1]
        input_train_r[idx,] = train_triple[idx][2]

    input_test_h = np.zeros((len(test_triple), 1)).astype('int32')  # 为测试数据创建的。这些矩阵存储每个训练三元组的头实体、尾实体和关系。
    input_test_t = np.zeros((len(test_triple), 1)).astype('int32')
    input_test_r = np.zeros((len(test_triple), 1)).astype('int32')
    for idx, tri in enumerate(test_triple):
        input_test_h[idx,] = tri[0]
        input_test_t[idx,] = tri[1]
        input_test_r[idx,] = tri[2]
    print(modelname)

    model = SelectModel(modelname, entvocabsize=len(ent_vocab), relvocabsize=len(rel_vocab),
                        ent2vec=entity2vec, rel2vec=relation2vec, input_path_length=max_p,
                        ent_emd_dim=entity2vec_dim, rel_emd_dim=relation2vec_dim)  # 该函数似乎是自定义模型选择函数
    model.load_state_dict(torch.load(modelfile))  # 使用 PyTorch 的方法加载权重

    #model.load_weights(modelfile)  # 加载预先训练的模型以进行推理。
    # nnmodel = load_model(lstm_modelfile)

    acc = test_model(model,
                     input_test_h, input_test_t, input_test_r,
                     test_path_h, test_path_t, test_path_r,
                     test_path2_h, test_path2_t, test_path2_r,
                     test_path3_h, test_path3_t, test_path3_r,
                     test_transE, test_rrank, test_confidence, resultfile)



    # acc = test_model(model,
    #               input_train_h, input_train_t, input_train_r,
    #               train_path_h, train_path_t, train_path_r,
    #               train_path2_h, train_path2_t, train_path2_r,
    #               train_path3_h, train_path3_t, train_path3_r,
    #               train_transE, train_rrank, train_confidence, resultfile)   #调用test_model带有多个参数的函数并将其返回值分配给acc变量
    # print(acc)

    # test_model_PR(model,
    #               input_test_h, input_test_t, input_test_r,
    #               test_path_h, test_path_t, test_path_r,
    #               test_path2_h, test_path2_t, test_path2_r,
    #               test_path3_h, test_path3_t, test_path3_r,
    #               test_transE, test_rrank, test_confidence, resultfile)
    # print("PR",acc)

    # test_model_PR(model,
    #                  input_train_h, input_train_t, input_train_r,
    #               train_path_h, train_path_t, train_path_r,
    #               train_path2_h, train_path2_t, train_path2_r,
    #               train_path3_h, train_path3_t, train_path3_r,
    #               train_transE, train_rrank, train_confidence, resultfile)

    #test_model_linkPrediction(model, datafile, entityRank,input_test_h, input_test_t, input_test_r)
    # test_model_load(model, ent_idex_word, rel_idex_word, test_triple, test_path_h, test_path_t, test_path_r, test_transE, test_rrank, test_confidence, resultfile)

    avg_confidence=test_model_load(model, ent_idex_word, rel_idex_word, test_triple,
                    input_test_h, input_test_t, input_test_r,
                    test_path_h, test_path_t, test_path_r,
                    test_path2_h, test_path2_t, test_path2_r,
                    test_path3_h, test_path3_t, test_path3_r,
                    test_transE, test_rrank, train_confidence, resultfile)
    # acc = test_model(model,
    #                  test_path_h_KGC_hr_, test_path_t_KGC_hr_, test_path_r_KGC_hr_,
    #                  test_transE_hr_, test_rrank_KGC_hr_,
    #                  test_confidence_KGC_hr_, resultfile)
    #print('hr_ acc ... ', acc)

    # test_model_load(model, ent_idex_word, rel_idex_word,
    #                 test_triple_KGC_hr_,
    #                 test_path_h_KGC_hr_, test_path_t_KGC_hr_, test_path_r_KGC_hr_,
    #                 test_transE_hr_, test_rrank_KGC_hr_,
    #                 resultfile+'_hr__conf.txt')
    #
    # acc = test_model(model,
    #                  test_path_h_KGC_h_t, test_path_t_KGC_h_t, test_path_r_KGC_h_t,
    #                  test_transE_h_t, test_rrank_KGC_h_t,
    #                  test_confidence_KGC_h_t, resultfile)
    # print('h_t acc ... ', acc)
    # test_model_load(model, ent_idex_word, rel_idex_word,
    #                 test_triple_KGC_h_t,
    #                 test_path_h_KGC_h_t, test_path_t_KGC_h_t, test_path_r_KGC_h_t,
    #                 test_transE_h_t, test_rrank_KGC_h_t,
    #                 resultfile+'_h_t_conf.txt')
    #
    # acc = test_model(model,
    #                  test_path_h_KGC__rt, test_path_t_KGC__rt, test_path_r_KGC__rt,
    #                  test_transE__rt, test_rrank_KGC__rt,
    #                  test_confidence_KGC__rt, resultfile)
    # print('_rt acc ... ', acc)
    # test_model_load(model, ent_idex_word, rel_idex_word,
    #                 test_triple_KGC__rt,
    #                 test_path_h_KGC__rt, test_path_t_KGC__rt, test_path_r_KGC__rt,
    #                 test_transE__rt, test_rrank_KGC__rt,
    #                 resultfile+'__rt_conf.txt')


def get_goldtriples():
    path = "E:/TTMF/data/Add_data/golddataset/"  # 设置为正确三元组文件所在的目录路径。
    goldtriples = []

    files = os.listdir(path)
    for file in files:
        # print(file)
        fo = open(path + file, 'r')
        lines = fo.readlines()
        for line in lines:
            nodes = line.rstrip('\n').split('\t')
            goldtriples.append((int(nodes[0]), int(nodes[1]), int(nodes[2])))
        fo.close()
    return goldtriples


if __name__ == "__main__":

    modelname = 'TRG-Trust'  # 模型名称

    file_data = "E:/TTMF/data/Drive"

    entity2idfile = file_data + "/entity2id.txt"  # 包含有关实体及其相应 ID 的信息的文件的路径
    relation2idfile = file_data + "/relation2id.txt"

    entity2vecfile = file_data + "/FB15K_PTransE_Entity2Vec_100.txt"  # 该文件存储知识图中实体的预训练嵌入。嵌入表示连续向量空间中实体的向量表示，可用于各种机器学习任务，包括知识图补全。
    relation2vecfile = file_data + "/FB15K_PTransE_Relation2Vec_100.txt"

    trainfile = file_data + "/KBE/datasets/conf_train2id.txt"  # 其中包含知识图完成任务的训练数据。该数据通常包括用于训练模型的实体和关系的三元组。
    testfile = file_data + "/KBE/datasets/conf_test2id.txt"  # 该路径指向包含知识图谱完成任务的测试数据的文件。该数据用于评估模型的性能。
    testfile_KGC_h_t = file_data + "/KBC/h_t.txt"  # 这些路径用于加载额外的测试数据以完成知识图谱，与 相比，其结构可能不同或具有不同的评估标准testfile。
    testfile_KGC_hr_ = file_data + "/KBC/hr_.txt"
    testfile_KGC__rt = file_data + "/KBC/_rt.txt"

    path_file = file_data + "/Path_4/"
    entityRank = file_data + "/ResourceRank_4/"  # 包含与知识图中实体排名相关的数据或资源的目录的路径。

    datafile = "./model/data2_TransE.pkl"  # 该路径指向存储知识图谱补全任务预处理数据的文件
    datafile_test = "./model/data2_TransE_test.pkl"  # 该路径指向存储知识图谱补全任务预处理数据的文件
    modelfile = "./model/model2_TransE.h5"  # 此路径用于指定保存训练模型的位置
    resultdir = "./data/result/"  # 存储知识图谱完成任务结果的基目录
    resultdir = "./data/result/Model_train_model_TransE_---"

    batch_size = 64  # 指定训练和测试的批量大小。
    retrain = False   # 个布尔变量（设置为False）指示模型是否应该重新训练。
    Test = True # 个布尔变量（设置为True）指示脚本是否用于测试。
    valid = False  # 验证相关的布尔变量
    Label = False
    datafile_Rebuild =False
    if datafile_Rebuild:  # 是否重新生成datafile，目的是当改变数据集时结果文件会发生，而不是沿用上一次的data
        print("datafile_start")
        get_data(entity2idfile=entity2idfile, relation2idfile=relation2idfile,
                 entity2vecfile=entity2vecfile, relation2vecfile=relation2vecfile, w2v_k=100,
                 trainfile=trainfile, testfile=testfile,
                 testfile_KGC_h_t=testfile_KGC_h_t,
                 testfile_KGC_hr_=testfile_KGC_hr_,
                 testfile_KGC__rt=testfile_KGC__rt,
                 path_file=path_file, max_p=15,
                 entityRank=entityRank,
                 datafile=datafile)
    else:
        if not os.path.exists(datafile):  # 检查预处理数据和训练模型是否已经存在
            print("Precess data....")
            get_data(entity2idfile=entity2idfile, relation2idfile=relation2idfile,
                     entity2vecfile=entity2vecfile, relation2vecfile=relation2vecfile, w2v_k=100,
                     trainfile=trainfile, testfile=testfile,
                     testfile_KGC_h_t=testfile_KGC_h_t,
                     testfile_KGC_hr_=testfile_KGC_hr_,
                     testfile_KGC__rt=testfile_KGC__rt,
                     path_file=path_file, max_p=15,
                     entityRank=entityRank,
                     datafile=datafile)
    '''调用了名为 get_data 的函数，传递了一系列参数，这个函数的作用是执行数据预处理。这个函数将加载实体和关系的信息、加载嵌入向量、加载训练和测试数据等，然后对这些数据进行处理。
    最终，生成的数据被保存到 datafile 文件中。'''
    if not os.path.exists(modelfile):
        print("data has extisted: " + datafile)
        print("Training model....")
        print(modelfile)
        train_model(modelname, datafile, modelfile, resultdir,
                    npochos=200, batch_size=batch_size, retrain=False)
    else:
        if retrain:
            print("ReTraining EE model....")
            train_model(modelname, datafile, modelfile, resultdir,
                        npochos=200, batch_size=batch_size, retrain=retrain)

    if Test:
        print("test start model")

        #对测试数据进行预处理，以满足实时的处理

        # get_data_test(entity2idfile=entity2idfile, relation2idfile=relation2idfile,
        #          entity2vecfile=entity2vecfile, relation2vecfile=relation2vecfile, w2v_k=100,
        #          trainfile=trainfile, testfile=testfile,
        #          testfile_KGC_h_t=testfile_KGC_h_t,
        #          testfile_KGC_hr_=testfile_KGC_hr_,
        #          testfile_KGC__rt=testfile_KGC__rt,
        #          path_file=path_file, max_p=15,
        #          entityRank=entityRank,
        #          datafile_test=datafile_test)

        infer_model(modelname, entityRank, datafile,datafile_test, modelfile, resultdir, batch_size=batch_size)
