#include <set>
#include <vector>
#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <array>

#define max_beam 16

struct BeamResult
{
    std::vector<int> word_ids_;      // 存储生成的词ID序列
    std::vector<int> beam_ids_;      // 存储对应路径的束ID
    std::vector<float> score_;       // 存储每一步的得分
    float acc_score_ = 0.f;          // 累积得分
    float penalty_score_ = 0.f;      // 带惩罚的得分
    
    void clear()
    {
        word_ids_.clear();           // 清空词ID序列
        beam_ids_.clear();           // 清空束ID序列
        score_.clear();              // 清空得分序列
        acc_score_ = 0.f;            // 重置累积得分
        penalty_score_ = 0.f;        // 重置惩罚得分
    }
};


typedef std::vector<BeamResult> BeamResultArr;  // 定义束结果数组类型
static const BeamResult kEmptyBeamResult;       // 定义空的束结果常量

class BeamSearch
{ 
protected:
    bool inited_ = false;            // 初始化标志
    int max_batch_ = 0;              // 最大批次数
    int max_len_ = 256;              // 最大长度
    int batch_ = 32;                 // 批次大小
    int beam_ = 4;                   // 束宽
    int eos_id_ = 2;                 // 结束标记ID
    
    float* acc_score_ = nullptr;     // 累积得分数组
    float* best_beams_ = nullptr;    // 最优束索引数组
    float* best_words_ = nullptr;    // 最优词索引数组
    
    float* max_val_tmp_ = nullptr;   // 临时存储最大值
    float* max_pos_tmp_ = nullptr;   // 临时存储最大值位置
    
    int* mask_ = nullptr;            // 掩码数组，0表示启用，1表示禁用
    
    std::vector<int> batch_finish_;  // 每个批次是否完成的标记
    std::vector<BeamResultArr> finish_result_;  // 已完成的结果
    std::vector<BeamResultArr> active_result_;  // 活跃的结果
    bool early_stop_ = true;         // 是否提前停止
public:
    BeamSearch(){}

    ~BeamSearch(){
        if(inited_){
            this->fini();
        }
    }

    // 1,beam_size,256,2
    // batch = 1，2 beam ,最大长度256  .eos在vob中索引2
    virtual void init(int max_batch, int beam, int max_len, int eos_id) {
        this->fini();
        beam_ = beam;
        eos_id_ = eos_id;
        max_len_ = max_len;
        batch_ = max_batch;
        max_batch_ = max_batch;

        best_words_ = new float[max_batch_*2*beam_];
        best_beams_ = new float[max_batch_*2*beam_];
        acc_score_ = new float[max_batch_*2*beam_];

        max_val_tmp_ = new float[batch_*beam_*2*beam_];// 2个beam。一个beam取topn(topn = 2*beam)
        max_pos_tmp_ = new float[batch_*beam_*2*beam_];

        mask_ = new int[max_batch_*beam_];

        finish_result_.resize(max_batch_);
        active_result_.resize(max_batch_);
        active_result_[0].resize(beam_);
        batch_finish_.resize(max_batch_);

        inited_ = true;
    }

    virtual void fini() {
        if (!inited_) return;

        delete [] best_words_;
        delete [] best_beams_;
        delete [] acc_score_;

        delete [] max_val_tmp_;
        delete [] max_pos_tmp_;

        delete [] mask_;

        inited_ = false;
    }
    virtual void reset(int batch, bool is_reset = true) {
        batch_ = batch;
        if (is_reset) {
            for (int i = 0; i < batch_; ++i) {
                this->reset_ibatch(i);
            }
        }
    }

    virtual void reset_ibatch(int i) {
        for (int b = 0; b < beam_; b++) {
            acc_score_[i*beam_+b] = -1e20; // -inf
            mask_[i*beam_+b] = 1;
            best_words_[i*beam_+b] = 0;
        }
        mask_[i*beam_] = 0;
        acc_score_[i*beam_] = 0;

        finish_result_[i].clear();
        active_result_[i].clear();
        active_result_[i].resize(beam_);
        batch_finish_[i] = 0;
    }

    virtual void search_preprocess() {
        return; // Doing nothing
    }

    virtual void search_postprocess() {
        update_result();
    }

    // input: batch*beam*vocab_size
    virtual void search(float* prob, int vocab_size) {
        this->search_preprocess();
        // 概率矩阵中找top-n个最大值
        topn_cpu(prob, mask_, batch_*beam_, vocab_size ,vocab_size, max_val_tmp_, max_pos_tmp_, 2*beam_);
        // 累加得分
        // 选择topk
        acc_cpu(max_val_tmp_, mask_, batch_*beam_, 2*beam_, acc_score_);
        beam_cpu(max_val_tmp_, max_pos_tmp_, best_beams_, best_words_, acc_score_, batch_, beam_, 2*beam_);

        this->search_postprocess();
    }

    virtual int *batch_finish() {
        return batch_finish_.data();
    }

    virtual float *best_words() { return best_words_; }
    virtual float *best_beams() { return best_beams_; }

    virtual const BeamResult &get_result(int ibatch, int ibeam) {
        if (batch_finish_[ibatch]) {
            return finish_result_[ibatch][ibeam];
        }
        return active_result_[ibatch][ibeam];
    }

    virtual std::vector<int> back_trace(int *data, int ibatch, int ibeam) {
        std::vector<int> output;
        if (batch_finish_[ibatch]) {
            auto &result_beams = get_result(ibatch, ibeam).beam_ids_;
            output.resize(result_beams.size());
            int *data_i = data;
            for (size_t i = 0; i < result_beams.size(); ++i) {
                output[i] = data_i[result_beams[i]];
            }
            data_i += beam_;
        }
        return std::move(output);
    }

    virtual void rearrange(void *src, void *dst, int row_bytes) {
        for (int i = 0; i < batch_; ++i) {
            for (int b = 0; b < beam_; ++b) {
                int ibeam = best_beams_[i*beam_+b];
                printf("beam_beams_[batch %d ][beam_id %d ] = %d\n", i, b, ibeam);
                memcpy((int8_t*)dst+(i*beam_+b)*row_bytes, (int8_t*)src+(i*beam_+ibeam)*row_bytes, row_bytes);
            }
        }
        return;
    }

    virtual float penalty(float score, float length) {
        return score / length;
    }

    //处理获取到的best_words_ best_beams_ 
    //维护 finish_result_ (已完成) 和 active_result_ (只有 beam个结果，top4取top2)
    //按惩罚得分排序完成结果
    virtual void update_result() {
        int beam2 = 2*beam_;

        for (int i = 0; i < batch_; ++i) {
            if (batch_finish_[i]) continue;
            BeamResultArr temp_result = active_result_[i];

            int num_active = 0; 
            for (int j = 0; j < beam2; ++j) {
                int wordid = best_words_[i*beam2+j];
                float beamid = best_beams_[i*beam2+j];
                float acc_score = acc_score_[i*beam2+j];
                int w = (int)wordid;
                int b = (int)beamid;
                if (w != eos_id_) {
                    int k = i*beam_+num_active; //在数组中的位置
                    best_words_[k] = wordid+0.01f;
                    best_beams_[k] = beamid;
                    acc_score_[k] = acc_score;
                    if (num_active != b) {//复制路径历史
                        active_result_[i][num_active] = temp_result[b];
                    }
                    auto & result = active_result_[i][num_active];
                    result.word_ids_.push_back(w);
                    result.beam_ids_.push_back(b);
                    result.score_.push_back(acc_score - result.acc_score_);
                    result.acc_score_ = acc_score;
                    ++num_active;
                    if (num_active >= beam_) break;
                } else {
                    if (j >= beam_) continue;
                    float pen_score = penalty(acc_score, temp_result[b].beam_ids_.size()+1);
                    if (finish_result_[i].size() < (size_t)beam_) {
                        finish_result_[i].push_back(temp_result[b]);
                    } else {
                        float worst_penalty_score = finish_result_[i].rbegin()->penalty_score_;
                        if (early_stop_ && pen_score < worst_penalty_score) { continue; }
                        finish_result_[i][beam_-1] = temp_result[b];
                    }
                    auto & result = finish_result_[i][finish_result_[i].size()-1];
                    result.beam_ids_.push_back(b);
                    result.score_.push_back(acc_score - result.acc_score_);
                    result.acc_score_ = acc_score;
                    result.penalty_score_ = penalty(acc_score, result.beam_ids_.size());
                    if (finish_result_[i].size() == (size_t)beam_) { sort_result(i); }
                }
            }
            // 结果重排
            // 使得best_beams_ 递增排序
            for (int j = 0; j < beam_; ++j) {
                if ((int)best_beams_[i*beam_+j] == j) continue;
                for (int k = 0; k < beam_; ++k) {
                    if ((int)best_beams_[i*beam_+k] == j) {
                        if (j == k) {
                            std::swap(best_beams_[i*beam_+k], best_beams_[i*beam_+j]);
                            std::swap(best_words_[i*beam_+k], best_words_[i*beam_+j]);
                            std::swap(acc_score_[i*beam_+k], acc_score_[i*beam_+j]);
                            std::swap(active_result_[i][k], active_result_[i][j]);
                        }
                        break;
                    }
                }
            }
        }
        for (int i = 0; i < batch_; ++i) {
            if (batch_finish_[i]) continue;
            if (finish_result_[i].size() == (size_t)beam_) {
                if (!early_stop_) {
                    int length = active_result_[i][0].word_ids_.size();
                    float max_score = active_result_[i][0].acc_score_;
                    float max_score_p = penalty(max_score, length);
                    if (max_score_p < finish_result_[i][beam_-1].penalty_score_) {
                        batch_finish_[i] = true;
                    }
                } 
                else {
                    batch_finish_[i] = true;
                }
            } else {
                int length = active_result_[i][0].word_ids_.size();
                if (length >= max_len_) {
                    int num_addition = beam_ - finish_result_[i].size();
                    for (int k = 0; k < num_addition; ++k) {
                        finish_result_[i].push_back(active_result_[i][k]);
                        auto &result = finish_result_[i][finish_result_[i].size()-1];
                        result.penalty_score_ = penalty(result.acc_score_, length)-1.f;
                    }
                    sort_result(i);
                    batch_finish_[i] = true;
                }
            }

            if (batch_finish_[i]) {
                for (int j = 0; j < beam_; ++j) {
                    mask_[i*beam_+j] = 1;
                }
            } else {
                for (int j = 0; j < beam_; ++j) {
                    mask_[i*beam_+j] = 0;
                }
            }
        }
    }

private:
    void sort_result(int i) {
        std::sort(finish_result_[i].begin(), finish_result_[i].end(),
                [](const BeamResult &l, const BeamResult &r) {
                    return l.penalty_score_ > r.penalty_score_;
                });
    };
};


// topk 大小为2*beam_
// data [2,1,20000]大小的prob_maxtrix
// topn_cpu(prob, mask_, 2, 20000, 20000, max_val_tmp_, max_pos_tmp_, 2*beam_);
static void topn_cpu(float* data, int* row_finish, int num_rows, int stride, int num_cols, float* maxVal, float* maxPos, int topn) {
    for (int irow = 0; irow < num_rows; ++irow) { //每行维护topn列表,这里是4
        if (row_finish[irow] != 0) continue;
        float* topn_value_arr = maxVal + topn*irow; 
        float* topn_pos_arr = maxPos + topn*irow;
        float* data_ = data + stride*irow;

        topn_value_arr[0] = data_[0]; //初始默认idx=0最大
        topn_pos_arr[0] = 0;
        // 前 topn 个元素初始化 value[topn] pos[topn] 
        for (int i = 1; i < topn; ++i) {
            float v = data_[i];
            int iv = i;
            // 往前遍历找插入位置
            while(iv > 0 && v > topn_value_arr[iv-1]) {
                iv--;
            }
            // 将找到插入位置之后的所有元素向后移动一位
            for (int j = i; j > iv; --j) {
                topn_value_arr[j] = topn_value_arr[j-1];
                topn_pos_arr[j] = topn_pos_arr[j-1];
            }

            // 插入元素
            topn_value_arr[iv] = v;
            topn_pos_arr[iv] = i;
        }
        // 遍历后续data 更新value[topn] pos[topn]
        for (int i = topn; i < num_cols; ++i) {
            float v = data_[i];
            if (v > topn_value_arr[topn-1]) { // 只与最小值比较
                int iv = topn-1;
                while(iv > 0 && v > topn_value_arr[iv-1]) { 
                    iv--;
                }
                for (int j = topn-1; j > iv; --j) {
                    topn_value_arr[j] = topn_value_arr[j-1];
                    topn_pos_arr[j] = topn_pos_arr[j-1];
                }
                topn_value_arr[iv] = v;
                topn_pos_arr[iv] = i;
            }
        }
    }
}
// 每个[batch][beam]的累计得分 累加到 topn-cpu得到的probs上
// acc_cpu(max_val_tmp_, mask_, batch_*beam_, 2*beam_, acc_score_);
// row_finish = 1 已完成，算法仍会遍历所有 2*beam_ 个候选,保留 prob_i[0]每行至少有一个有效值，便于后续处理
static void acc_cpu(float *prob, int *row_finish, int num_rows, int num_cols, float *cur_cost) {
    for (int i = 0; i < num_rows; ++i) {
        float *prob_i = prob + i * num_cols;
        if (row_finish[i] != 0) {
            for (int j = 1; j < num_cols; ++j) {
                prob_i[j] = -1e20;
            }
        } else {
            for (int j = 0; j < num_cols; ++j) {
                prob_i[j] += cur_cost[i];
            }
        }
    }
}

//从 2*beam_ 个候选项中选出 beam_ 个最佳组合（束ID和词ID)
// beam_cpu(max_val_tmp_, max_pos_tmp_, best_beams_, best_words_, acc_score_, batch, 2, 2*2);
// 一个seq对应2个beam，每个beam topn = 4 个候选
// beam行topn列数据
// 比如prob为[0.8, 0.6, 0.4, 0.2 0.9, 0.5, 0.3, 0.1]
// beam中交替选择rows个结果 ，pos_beams[i]为第i个beam的prob偏移

static void beam_cpu(float *prob, float *pos, // input
                     float *best_beams, float *best_words, float *best_scores, // output
                     int batch, int beam, int rows) {
    for (int i = 0; i < batch; ++i) {
        float *prob_i = prob + i * beam * rows;
        float *pos_i = pos + i * beam * rows;
        float *best_beams_i = best_beams + i * rows;// beam_id
        float *best_words_i = best_words + i * rows;// pos
        float *best_scores_i = best_scores + i * rows; //value
        int pos_beams[MAX_BEAM];

        for (int j = 0; j < beam; ++j) {
            pos_beams[j] = j * rows; // pos_beams = [0, 4] 指beam起始位置
        }

        int itop = 0;
        while (itop < rows) {
            int max_beam = 0; //初始化beam0的值为max_beam
            int max_pos = pos_beams[0];
            float max_val = prob_i[max_pos];
            for (int j = 1; j < beam; ++j) {//比较beam0和beam1的top1
                if (prob_i[pos_beams[j]] > max_val) {
                    max_beam = j;
                    max_pos = pos_beams[j];
                    max_val = prob_i[max_pos];
                }
            }

            best_beams_i[itop] = max_beam;
            best_words_i[itop] = pos_i[max_pos];
            best_scores_i[itop] = max_val;
            ++pos_beams[max_beam];
            ++itop;
        }
    }
}