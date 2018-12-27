## English Article Summary 2018-12

#### 1.preprocess

clean() 删去无用字符并分词，prepare() 将数据保存为 (text1, text2) 格式

#### 2.represent

add_flag() 添加控制符，shift() 对 text2 分别删去 bos、eos 得到 sent2、label

tokenize() 分别通过 sent1、flag_text2 建立词索引，构造 embed_mat

align_sent() 填充或截取为定长序列，align_label() 再构造未录词到 text1 的指针

#### 3.build

通过 rnn 的 ptr 构建摘要模型，连接语境向量 c 与解码器词特征 h2 得到 s2



#### 4.summary

先对输入进行编码、再通过搜索进行解码，check() 忽略无效词，plot_att() 可视化