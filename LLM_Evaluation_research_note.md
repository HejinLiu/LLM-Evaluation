# LLM-Evaluation-Project-Note

## 1. 概述

本项目使用Python复现了CP和MCP两种评估方法。分别在数据集 ARC-Challenge 和 CommonQA数据集的测试集上，针对 Qwen2-Instruct 的0.5B、1.5B两种大小的模型，在提示中样本数量k={0,1,2,5}的k-shot实验进行了评估。

| 参数                    | 取值                                     |
| ----------------------- | ---------------------------------------- |
| 数据集                  | ARC-Challenge, CommonQA                  |
| 模型                    | Qwen2-0.5B-Instruct, Qwen2-1.5B-Instruct |
| k-shot 提示中样本数量 k | 0, 1, 2, 5                               |

## 2. 实验与结果分析

以准确率ACC作为评价指标，统计了在不同设置下的评估结果如下：

![实验结果表](/figures/exp_results.png)

实验结果也存储在 `aggregate_results.xlsx` 文件中。

### 实验结果分析

1. **MCP 方法性能提升显著**  
   MCP 方法在零样本测试中表现出色，相比 CP 方法，在大部分提示方法下在准确率上提升了 10%以上，尤其是在CommonQA数据集上，涨点可以达到 15 ~ 55%，这与相关文献中的结果一致。
   
   但是在模型参数量较小的时候（0.5B），在ARC-Challenge数据集的部分设置上（如1-shot、2-shot），MCP表现不佳。
   
2. **提示样本数量k的影响**  
   随着提示样本数量 k 的增加，CP_Raw 和 CP_LN 的表现基本稳定且略微提升，但 CP_UN 和 MCP 的性能未必严格随着 k 的增加而提升。
   
3. **MCP在零样本测试中展现出较大的优势，而提升样本量k-shot的k值对MCP影响不大，确可以使CP效果得到较大提升。**

4. **实验鲁棒性**

   本实验鲁棒性其实不强，我重复做了三次相同配置的实验（实验配置为：MCP方法、0.5B模型、2-shot），其准确率分别为：

   ```
   0.30, 0.32, 0.29
   ```

   绝对误差达到3%，相对误差达到自身正确率的10%。这证明LLM的输出还是不太稳定的，其实本实验应该多次试验取平均值。

> **补充：CP 任务下归一化方法的介绍：**  
> CP 方法选择答案时，LLM 根据生成每个答案的概率进行评分，选择生成概率最高的选项。为减轻生成概率受 token 频率或序列长度的影响，研究中使用了两种归一化方法：
>
> - **Raw**: 对生成概率不作处理，直接选择概率最高的答案选项。
> - **LN**: 通过 n 次方根归一化序列概率，即 $$ P ( x _ { 1 } , x _ { 2 } , \ldots , x _ { n } ) = \sqrt [ n ] { \prod _ { i = 1 } ^ { n } P ( x _ { i } ) } $$。
> - **UN**: 答案概率通过无条件概率归一化，即 P(completion|context)P(completion|answer−context) \frac { P ( c o m p l e t i o n | c o n t e x t ) } { P ( c o m p l e t i o n | a n s w e r - c o n t e x t ) } 。

### 问题

1. 在CP方法的归一化中，我发现两种归一化（LN、UN）对于性能的提升并不明显，这和原论文中的结果是相悖的，这一点现在使我有些困惑。下图是原论文中四种方法的指标，可以看出两种归一化方法对CP的提升虽然不多，但还是有一些的基本能达到2~7%，这和我的实验结果有些出入。

   ![image-20240703185213600](/figures/results_origin.png)

2. CommonQA数据集在0.5B模型上的表现略微奇怪，只有MCP在0-shot下准确率表现达到40%，其他设置表现均不到20%（还不如随机答案的正确率高），但是这个问题在1.5B模型上有所缓解。我再三检查代码确认代码过程无误。

### 实验设置

为平衡评估时间和数据量，实验选择了 ARC-Challenge 的测试集（1176 条问答）和 CommomQA 测试集前 1000 条问答（总计 9900 余条）。评估环境为单卡 NVIDIA RTX 4090，评估速度一般在 1.5 ~ 2.5 条每秒。



## 3. 关于不同评估方式的思考

### 1. 结构化数据输出格式对 CP 的性能限制

CP评估方法中，考虑LLM输出符合条件的问答字符串`S`的概率，字符串`S`格式如下：

```
Question: {Q}
Answer: {A}
```

其中，`Q`是问题的字符串，比如`Q = "Greenhouses are great for plants like"`，`A`是答案字符串，比如`A = "French beans"`。`S`其实算一个符合上述模板的**半结构化**字符串。

但是，如果LLM没有接受过专门的训练，反而其预训练中包含的多是自然语言的文本，那么他生成这样半结构化字符串的概率会受到影响。

下面举例说明：

> 假设模型在预训练语料中，有如下数据：
>
> ```json
> {"sentence1": "Snow forms when atmospheric temperatures are at or below freezing (0°C or 32°F), and there is a minimum amount of moisture in the air. "}
> {"sentence2": "In chemistry class, the teacher asked us what water would become if it solidifies, and I told the teacher the answer: ice."}
> ```
>
> 那么如果对模型进行提问：
>
> ```
> What substance falls from the sky in winter and covers the ground with a white blanket, but is not a solid block?
> ```
>
> 模型会发现需要回答一个形如“Answer: xxxx”的语句，虽然他在训练的时候接受过类似sentence1的信息，得知了雪的形成条件，但是模型发现出现了形如“Answer：xxxx”的地方只有sentence2，而且从词向量的角度思考，ice和snow的词向量或许很接近，至少在当前环境下是接近的，那么LLM很有可能说出“Answer: ice”这个错误答案

### 2. MCP的缺点：需要使用 System Prompt约束其回答结构

我使用了如下的System Prompt来约束MCP的输出，使MCP提示下的模型的输出尽量仅包含一个字母：

```json
{"role": "system",
 "content": "Below, I will give you a question. Please choose the correct answer from this question and output the correct option letter. Your answer should only contain one letter"}
```

但是我发现模型并不一定能按照要求回答，于是我统计了0.5B模型在ACR数据集上输出不符合约束的概率：

```
在1172条数据中，0.5B模型在ACR数据集上的输出中包含不符合约束的输出的数量：
0-shot: 100条
1-shot: 170条
2-shot: 230条
5-shot: 124条
```

总体不符合约束的概率约为13%。

所以，从提示词工程的角度，MCP的潜力依然没有被严格挖掘。可以考虑以下两个改进措施：

1. **在System Prompt中加入范例的问答样本**

   在System Prompt中加入一个范例的问答样本，加强提示的约束力。缺点：这样就不能说自己做的事0-shot实验了，而且也不太清楚在System Prompt中加入一个范例的问答样本和1-shot对比，谁的约束力更强。我对不同shot数的异常回答做了统计，结果发现shot的增多并不一定意味着模型的输出会越来越“听话”，所以这种方法的可行性有待商榷。
   
   从上述数据统计可以看到，0样本测试下的错误回答反而更少，2样本反而多，这说明增加提示数量并不一定会更加有效地规范模型输出。但是这与我们的直觉相悖，其原因暂时不得而知。可能是和模型性能有关，小样本模型记忆力、推理能力差，shot多了模型反而记不住东西了。
   
2. **正则匹配**

   通过正则提取，判断模型回答的答案究竟是什么。
   
   但是有可能出现LLM只回答了选项的文本内容，而没有回答选项标签的字母。甚至有可能LLM还在答案中解释了自己这样回答的原因，所以不可避免地会有一定的误判，或者有部分句子需要人为判断，增加判断的成本。

> 补充：在我自己本科期间的项目中，我们也考虑过在评估时的问题。
>
> 我们的项目是研究对LLM进行提示框架，诱导其进行情感分类。我们的最后一步提示词是：
> ```
> Based on the above reasons, how would you describe the sentiment polarity towards the
> sentence?
> ```
>
> 我们当时发现模型一般输出的第一句话是很“规矩”的，类似
>
> ```
> Based on the above reasons, the sentiment polarity of the sentence is poasitive...
> ```
>
> 但是模型也有可能说出一些其他的话，比如：
>
> ```
> The sentiment polarity of the sentence is satirical and gloomy...（使用了positive、negative、neutrual以外的词，虽然知道表达是消极的意思，但是无法直接捕捉情感）
> 
> The sentiment polarity of the sentence is mixed...（回答了mixed，不属于任何一类）
> 
> The sentiment polarity of the sentence is neutrual and a little bit positive...（说大体中性但是有点positive，模棱两可）
> ```
>
> 针对这些问题，我们想了两种方法，要么是加一步提示，"Please summarize your emotional analysis of this sentence as a word in "positive, negative, neutral"；要么是是使用正则匹配，提取第一个标点符号之前，negative、neutrual、positive的输出次数，统计大部分正常的语句判断结果，然后手动清理小部分的异常结果。最终我们选择了第二种方法，因为我们怕如果让模型在最后conclude一次，这一个环节会产生误差。

### 3. 关于coding经验

本次任务在coding中我也积累了一些经验，列举如下

- **LN 归一化计算错误**

  在进行LN归一化时，很容易一不小心把p(context)计算成模型生成最后一个token的概率，而不是生成完整的“Answer: {A}”字符串的概率。我一开始coding的时候这里写错了，后来才改过来了

  > 一开始的错误代码：
  >
  > ```python
  > logits = outputs.logits  # 提取 logits
  > last_token_logits = logits[:, -1, :]  # 获取最后一个 token 的 logits
  > prob = torch.softmax(last_token_logits, dim=-1)  # 计算 softmax 概率分布
  > answer_token_id = model_inputs.input_ids[0, -1].item()  # 获取目标 token 的 ID
  > score = prob[0, answer_token_id].item()  # 提取目标 token 的概率
  > ```
  >
  > 后来改正的代码：
  >
  > ```python
  > uncond_logits = uncond_outputs.logits
  > 
  > # Calculate the log probabilities for the entire sequence
  > log_probs = torch.log_softmax(uncond_logits, dim=-1)
  > input_ids = uncond_model_inputs['input_ids']
  > 
  > # We need to sum the log probabilities for all tokens in the answer
  > uncond_score = 0.0
  > for i in range(len(input_ids[0]) - 1):
  >     uncond_score += log_probs[0, i, input_ids[0, i + 1]].item()
  > 
  > uncond_score = np.exp(uncond_score)  # Convert log probabilities to actual probabilities
  > ```
  >
  > 在上面的代码中，`log_probs[0, i, input_ids[0, i + 1]]` 表示序列中第 i 个 token 生成第 i+1 个 token 的 log 概率，对这些 log 概率进行累加以获得整个字符串的 log 概率，累加后的 log 概率通过 `np.exp()` 转换为实际概率。

- **parquet格式的数据集处理**

  parquet格式的数据集是我第一次遇见，可以使用 Pandas 将 parquet 转换为 DataFrame 处理。

- **modelscope库**

  modelscope库是我第一次用，之前一直用transformers库，这个库似乎有对国内用户更友好的网络访问能力，下载和使用Qwen的LLM的时候很好用。

