# Awesome-CVPR2020-Image-Reconstruction[![Awesome](https://camo.githubusercontent.com/13c4e50d88df7178ae1882a203ed57b641674f94/68747470733a2f2f63646e2e7261776769742e636f6d2f73696e647265736f726875732f617765736f6d652f643733303566333864323966656437386661383536353265336136336531353464643865383832392f6d656469612f62616467652e737667)](https://github.com/sindresorhus/awesome)
A Collection of Papers and Codes for CVPR2020 Low Level Vision or Image Reconstruction

整理汇总了下今年CVPR图像重建(Image Reconstruction)/底层视觉(Low-Level Vision)相关的一些论文，包括超分辨率，图像恢复，去雨，去雾，去模糊，去噪等方向。大家如果觉得有帮助，欢迎star~~
- CVPR2020的所有论文：[http://openaccess.thecvf.com/CVPR2020.py](http://openaccess.thecvf.com/CVPR2020.py)
- CVPR2020Workshops：[http://openaccess.thecvf.com/CVPR2020_workshops/menu.py](http://openaccess.thecvf.com/CVPR2020_workshops/menu.py)
# 1.超分辨率
## 图像超分辨率
### PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models
- 论文：[https://arxiv.org/abs/2003.03808](https://arxiv.org/abs/2003.03808)
- 代码：[https://github.com/adamian98/pulse](https://github.com/adamian98/pulse)
- 解析：[杜克大学提出 AI 算法，拯救渣画质马赛克秒变高清](https://mp.weixin.qq.com/s?__biz=MzU3NTQ2NDIyOQ==&mid=2247493978&idx=1&sn=48a3a6c775ae490b2d6f90bcc607b075&chksm=fd201f10ca5796062a968040ed7e6bfebe8085b183b4ea0e1117c38ce6ec66c6d728ca483fba&mpshare=1&scene=23&srcid=&sharer_sharetime=1592366037552&sharer_shareid=e2c873c63ba339b9a32a1bfa460a6ebf#rd)
- 备注：自监督；GAN；放大像素64倍（暂时是最高倍数）；将生成HR图像对应的LR图像与原图（LR）对比，找到最接近的那张，并反推找到对应的HR图像
### Closed-Loop Matters: Dual Regression Networks for Single Image Super-Resolution
- 论文：[https://arxiv.org/abs/2003.07018](https://arxiv.org/abs/2003.07018)
- 代码：[https://github.com/guoyongcs/DRN](https://github.com/guoyongcs/DRN)
- 解析：
- 备注：DRN
### EventSR: From Asynchronous Events to Image Reconstruction, Restoration, and Super-Resolution via End-to-End Adversarial Learning
- 作者： Lin Wang, Tae-Kyun Kim, Kuk-Jin Yoon
- 单位：韩国科学技术院；伦敦帝国学院
- 论文：[http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_EventSR_From_Asynchronous_Events_to_Image_Reconstruction_Restoration_and_Super-Resolution_CVPR_2020_paper.pdf](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_EventSR_From_Asynchronous_Events_to_Image_Reconstruction_Restoration_and_Super-Resolution_CVPR_2020_paper.pdf)
- 视频 ：[https://www.youtube.com/watch?v=OShS_MwHecs](https://www.youtube.com/watch?v=OShS_MwHecs)
- 数据集： [https://github.com/wl082013/ESIM_dataset](https://github.com/wl082013/ESIM_dataset)
- 备注：图像重建、恢复、超分
### Unpaired Image Super-Resolution Using Pseudo-Supervision
- 论文：[https://arxiv.org/abs/2002.11397?context=eess](https://arxiv.org/abs/2002.11397?context=eess)
- 代码：
- 解析：[#每日五分钟一读#Image Super-Resolution](https://zhuanlan.zhihu.com/p/147171955)
- 备注：
### Correction Filter for Single Image Super-Resolution: Robustifying Off-the-Shelf Deep Super-Resolvers
- 作者： Shady Abu Hussein, Tom Tirer, Raja Giryes
- 论文：[https://arxiv.org/abs/1912.00157](https://arxiv.org/abs/1912.00157)
### Residual Feature Aggregation Network for Image Super-Resolution
- 论文：[http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Residual_Feature_Aggregation_Network_for_Image_Super-Resolution_CVPR_2020_paper.pdf](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Residual_Feature_Aggregation_Network_for_Image_Super-Resolution_CVPR_2020_paper.pdf)
- 代码：
- 解析：[超越RCAN，图像超分又一峰：RFANet](https://mp.weixin.qq.com/s/laZ_LH8tQ4PQaQXIyr38Bg)
- 备注：超越RCAN，图像超分又一峰：RFANet
### Deep Unfolding Network for Image Super-Resolution
- 论文：[https://arxiv.org/abs/2003.10428](https://arxiv.org/abs/2003.10428)
- 代码：[https://github.com/cszn/USRNet](https://github.com/cszn/USRNet)
- 解析：[CVPR2020：USRNet](https://mp.weixin.qq.com/s/0-Kjfi8RBuPb_KuB5-Khkg)
- 备注：USRNet
### Image Super-Resolution With Cross-Scale Non-Local Attention and Exhaustive Self-Exemplars Mining
- 论文：[https://arxiv.org/abs/2006.01424](https://arxiv.org/abs/2006.01424)
- 代码：[https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention](https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention)
### Learning Texture Transformer Network for Image Super-Resolution
- 论文：[https://arxiv.org/abs/2006.04139](https://arxiv.org/abs/2006.04139)
- 代码：[https://github.com/FuzhiYang/TTSR](https://github.com/FuzhiYang/TTSR)
- 备注：注意力机制
### Robust Reference-Based Super-Resolution With Similarity-Aware Deformable Convolution
- 论文：[http://openaccess.thecvf.com/content_CVPR_2020/html/Shim_Robust_Reference-Based_Super-Resolution_With_Similarity-Aware_Deformable_Convolution_CVPR_2020_paper.html](http://openaccess.thecvf.com/content_CVPR_2020/html/Shim_Robust_Reference-Based_Super-Resolution_With_Similarity-Aware_Deformable_Convolution_CVPR_2020_paper.html)
### Structure-Preserving Super Resolution with Gradient Guidance
- 论文：[https://arxiv.org/abs/2003.13063](https://arxiv.org/abs/2003.13063)
- 代码：[https://github.com/Maclory/Deep-Iterative-Collaboration](https://github.com/Maclory/Deep-Iterative-Collaboration)
- 解析：[CVPR2020丨SPSR：基于梯度指导的结构保留超分辨率方法](https://zhuanlan.zhihu.com/p/121721537)
- 备注：SPSR

### Unified Dynamic Convolutional Network for Super-Resolution With Variational Degradations
- 论文：[https://arxiv.org/abs/2004.06965](https://arxiv.org/abs/2004.06965)
- 代码：
- 解析：[UDVD：适用于可变降质类型的通用图像超分，附参考代码](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA%3D%3D&mid=2247494694&idx=1&sn=ed29071f700b129534beb649a04b3b97&scene=45#wechat_redirect)
- 备注：UDVD
### Perceptual Extreme Super Resolution Network with Receptive Field Block
- 论文：[https://arxiv.org/abs/2005.12597](https://arxiv.org/abs/2005.12597)
- 代码：
- 解析：[NTIRE2020冠军方案RFB-ESRGAN：带感受野模块的超分网络](https://mp.weixin.qq.com/s/nj_C_LXFpWQZASc4ITJmmA)
- 备注：NTIRE2020极限超分冠军方案RFB-ESRGAN；Workshops
### Real-World Super-Resolution via Kernel Estimation and Noise Injection
- 论文：[http://openaccess.thecvf.com/content_CVPRW_2020/html/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.html](http://openaccess.thecvf.com/content_CVPRW_2020/html/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.html)
- 代码：[https://github.com/jixiaozhong/RealSR](https://github.com/jixiaozhong/RealSR)
- 解析：
- 备注：NTIRE2020-RWSR超分双赛道冠军方案；Workshops
### Investigating Loss Functions for Extreme Super-Resolution
- 论文：[http://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Jo_Investigating_Loss_Functions_for_Extreme_Super-Resolution_CVPRW_2020_paper.pdf](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Jo_Investigating_Loss_Functions_for_Extreme_Super-Resolution_CVPRW_2020_paper.pdf)
- 代码：[https://github.com/kingsj0405/ciplab-NTIRE-2020](https://github.com/kingsj0405/ciplab-NTIRE-2020)
- 解析：
- 备注：NTIRE2020极限超分亚军方案CIPLab；Workshops
### Nested Scale-Editing for Conditional Image Synthesis
- 论文：[http://arxiv.org/abs/2006.02038](http://arxiv.org/abs/2006.02038)
- 备注：解耦表征、多模图像转换、超分、修复
### MSG-GAN: Multi-Scale Gradients for Generative Adversarial Networks
- 论文：[https://arxiv.org/abs/1903.06048v3](https://arxiv.org/abs/1903.06048v3)
- 代码：[https://github.com/akanimax/msg-stylegan-tf](https://github.com/akanimax/msg-stylegan-tf)
- 解析：[CVPR2020之MSG-GAN：简单有效的SOTA](https://mp.weixin.qq.com/s?__biz=MzU5MTgzNzE0MA==&mid=2247484246&idx=1&sn=3af777e066a3d4b6dcf1c4cf4856a671&chksm=fe29a1edc95e28fb85299ccf87a1819755f80243ff17f0057c425f3c4077e7e3b4d4a2035329&scene=21#wechat_redirect)
- 备注：NTIRE2020极限超分亚军方案CIPLab；Workshops
## 视频超分辨率
### TDAN: Temporally-Deformable Alignment Network for Video Super-Resolution
- 论文：[https://arxiv.org/abs/1812.02898](https://arxiv.org/abs/1812.02898)
- 代码：[https://github.com/YapengTian/TDAN-VSR-CVPR-2020](https://github.com/YapengTian/TDAN-VSR-CVPR-2020)
Demo Video：[https://www.youtube.com/watch?v=eZExENE50I0](https://www.youtube.com/watch?v=eZExENE50I0)
- 备注：首次将形变卷积用到视频超分领域；TDAN
### Zooming Slow-Mo: Fast and Accurate One-Stage Space-Time Video Super-Resolution
- 论文：[https://arxiv.org/abs/2002.11616](https://arxiv.org/abs/2002.11616)
- 代码：[https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020)
- 解析：[慢镜头变焦：视频超分辨率：CVPR2020论文解析](https://zhuanlan.zhihu.com/p/117341018)
### Video Super-Resolution With Temporal Group Attention
- 论文：[http://openaccess.thecvf.com/content_CVPR_2020/papers/Isobe_Video_Super-Resolution_With_Temporal_Group_Attention_CVPR_2020_paper.pdf](http://openaccess.thecvf.com/content_CVPR_2020/papers/Isobe_Video_Super-Resolution_With_Temporal_Group_Attention_CVPR_2020_paper.pdf)
### Space-Time-Aware Multi-Resolution Video Enhancement
- 主页：[https://alterzero.github.io/projects/STAR.html](https://alterzero.github.io/projects/STAR.html)
- 论文：[http://arxiv.org/abs/2003.13170](http://arxiv.org/abs/2003.13170)
- 代码：[https://github.com/alterzero/STARnet](https://github.com/alterzero/STARnet)
## 人脸超分辨率
### Learning to Have an Ear for Face Super-Resolution
- 论文：[https://arxiv.org/abs/1909.12780](https://arxiv.org/abs/1909.12780)
### Deep Face Super-Resolution With Iterative Collaboration Between Attentive Recovery and Landmark Estimation
- 论文：[https://arxiv.org/abs/1812.02898](https://arxiv.org/abs/1812.02898)
- 代码：[https://github.com/YapengTian/TDAN-VSR-CVPR-2020](https://github.com/YapengTian/TDAN-VSR-CVPR-2020)
## 深度图超分辨率
### Channel Attention Based Iterative Residual Learning for Depth Map Super-Resolution
- 论文：[https://arxiv.org/abs/2006.01469](https://arxiv.org/abs/2006.01469)
## 光场图像超分辨率
### Light Field Spatial Super-Resolution via Deep Combinatorial Geometry Embedding and Structural Consistency Regularization
- 论文：[https://arxiv.org/abs/2004.02215](https://arxiv.org/abs/2004.02215)
- 代码：[https://github.com/jingjin25/LFSSR-ATO](https://github.com/jingjin25/LFSSR-ATO)
## 高光谱图像超分辨率
### Unsupervised Adaptation Learning for Hyperspectral Imagery Super-Resolution
- 论文：[http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Unsupervised_Adaptation_Learning_for_Hyperspectral_Imagery_Super-Resolution_CVPR_2020_paper.pdf](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Unsupervised_Adaptation_Learning_for_Hyperspectral_Imagery_Super-Resolution_CVPR_2020_paper.pdf)
- 代码：[https://github.com/JiangtaoNie/UAL](https://github.com/JiangtaoNie/UAL)
## 零样本超分辨率
### Meta-Transfer Learning for Zero-Shot Super-Resolution
- 论文：[https://arxiv.org/abs/2002.12213](https://arxiv.org/abs/2002.12213)
- 代码：[https://github.com/JWSoh/MZSR](https://github.com/JWSoh/MZSR)
## 用于超分辨率的数据增广
### Rethinking Data Augmentation for Image Super-resolution: A Comprehensive Analysis and a New Strategy
- 论文：[http://openaccess.thecvf.com/content_CVPR_2020/html/Yoo_Rethinking_Data_Augmentation_for_Image_Super-resolution_A_Comprehensive_Analysis_and_CVPR_2020_paper.html](http://openaccess.thecvf.com/content_CVPR_2020/html/Yoo_Rethinking_Data_Augmentation_for_Image_Super-resolution_A_Comprehensive_Analysis_and_CVPR_2020_paper.html)
- 代码：[https://github.com/clovaai/cutblur](https://github.com/clovaai/cutblur)
## 超分辨率用于语义分割
### Dual Super-Resolution Learning for Semantic Segmentation
- 论文：[http://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Dual_Super-Resolution_Learning_for_Semantic_Segmentation_CVPR_2020_paper.html](http://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Dual_Super-Resolution_Learning_for_Semantic_Segmentation_CVPR_2020_paper.html)
- 代码：[https://github.com/wanglixilinx/DSRL](https://github.com/wanglixilinx/DSRL)
# 2.图像恢复
### Learning Invariant Representation for Unsupervised Image Restoration
- 论文：[https://arxiv.org/pdf/2003.12769.pdf](https://arxiv.org/pdf/2003.12769.pdf)
- 代码：[https://github.com/Wenchao-Du/LIR-for-Unsupervised-IR](https://github.com/Wenchao-Du/LIR-for-Unsupervised-IR)
### Contextual Residual Aggregation for Ultra High-Resolution Image Inpainting
- 论文：[https://arxiv.org/abs/2005.09704](https://arxiv.org/abs/2005.09704)
- 备注：超高分辨率图像修复、注意力机制
### UCTGAN: Diverse Image Inpainting based on Unsupervised Cross-Space
- 论文：[http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_UCTGAN_Diverse_Image_Inpainting_Based_on_Unsupervised_Cross-Space_Translation_CVPR_2020_paper.pdf](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_UCTGAN_Diverse_Image_Inpainting_Based_on_Unsupervised_Cross-Space_Translation_CVPR_2020_paper.pdf)
- 备注：图像修复、注意力机制
### Attentive Normalization for Conditional Image Generation
- 论文：[https://arxiv.org/abs/2004.03828](https://arxiv.org/abs/2004.03828)
- 备注：注意力机制、类条件图像生成、图像修复
# 3.去雨
### Deep Adversarial Decomposition: A Unified Framework for Separating Superimposed Images
- 论文：[http://openaccess.thecvf.com/content_CVPR_2020/html/Zou_Deep_Adversarial_Decomposition_A_Unified_Framework_for_Separating_Superimposed_Images_CVPR_2020_paper.html](http://openaccess.thecvf.com/content_CVPR_2020/html/Zou_Deep_Adversarial_Decomposition_A_Unified_Framework_for_Separating_Superimposed_Images_CVPR_2020_paper.html)
### Multi-Scale Progressive Fusion Network for Single Image Deraining
- 论文：[https://arxiv.org/abs/2003.10985](https://arxiv.org/abs/2003.10985)
- 代码：[https://github.com/kuihua/MSPFN](https://github.com/kuihua/MSPFN)
# 4.去雾
### Domain Adaptation for Image Dehazing
- 论文：[https://arxiv.org/abs/2005.04668](https://arxiv.org/abs/2005.04668)
### Multi-Scale Boosted Dehazing Network with Dense Feature Fusion
- 论文：[https://arxiv.org/abs/2004.13388](https://arxiv.org/abs/2004.13388)
- 代码：[https://github.com/BookerDeWitt/MSBDN-DFF](https://github.com/BookerDeWitt/MSBDN-DFF)
# 5.去模糊
## 视频去模糊
### Cascaded Deep Video Deblurring Using Temporal Sharpness Prior
- 主页：[https://csbhr.github.io/projects/cdvd-tsp/index.html](https://csbhr.github.io/projects/cdvd-tsp/index.html)
- 论文：[https://arxiv.org/abs/2004.02501](https://arxiv.org/abs/2004.02501)
- 代码：[https://github.com/csbhr/CDVD-TSP](https://github.com/csbhr/CDVD-TSP)
# 6.去噪
### A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising
- 论文：[https://arxiv.org/abs/2003.12751](https://arxiv.org/abs/2003.12751)
- 代码：[https://github.com/Vandermode/NoiseModel](https://github.com/Vandermode/NoiseModel)
### CycleISP: Real Image Restoration via Improved Data Synthesis
- 论文：[https://arxiv.org/abs/2003.07761](https://arxiv.org/abs/2003.07761)
- 代码：[https://github.com/swz30/CycleISP](https://github.com/swz30/CycleISP)
<font color=red size=5>未完待续~</font>
<div class="output_wrapper" id="output_wrapper_id" style="font-size: 16px; color: rgb(62, 62, 62); line-height: 1.6; word-spacing: 0px; letter-spacing: 0px; font-family: 'Helvetica Neue', Helvetica, 'Hiragino Sans GB', 'Microsoft YaHei', Arial, sans-serif;"><h1 id="h" style="color: inherit; line-height: inherit; padding: 0px; margin: 1.5em 0px; font-weight: bold; font-size: 1.6em;"><span style="font-size: inherit; color: inherit; line-height: inherit; margin: 0px; padding: 0px;">参考</span></h1>
<p style="font-size: inherit; color: inherit; line-height: inherit; padding: 0px; margin: 1.5em 0px;"><a href="https://mp.weixin.qq.com/s?__biz=MzU3NTQ2NDIyOQ==&amp;mid=2247493978&amp;idx=1&amp;sn=48a3a6c775ae490b2d6f90bcc607b075&amp;chksm=fd201f10ca5796062a968040ed7e6bfebe8085b183b4ea0e1117c38ce6ec66c6d728ca483fba&amp;mpshare=1&amp;scene=23&amp;srcid=&amp;sharer_sharetime=1592366037552&amp;sharer_shareid=e2c873c63ba339b9a32a1bfa460a6ebf#rd" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[1] 杜克大学提出 AI 算法，拯救渣画质马赛克秒变高清</a><br><a href="https://blog.csdn.net/moxibingdao/article/details/106726667" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[2] CVPR 2020 论文大盘点-超分辨率篇</a><br><a href="https://zhuanlan.zhihu.com/p/121721537" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[3] CVPR2020丨SPSR：基于梯度指导的结构保留超分辨率方法</a><br><a href="https://mp.weixin.qq.com/s/0-Kjfi8RBuPb_KuB5-Khkg" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[4] CVPR2020：USRNet</a><br><a href="https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA%3D%3D&amp;mid=2247494694&amp;idx=1&amp;sn=ed29071f700b129534beb649a04b3b97&amp;scene=45#wechat_redirect" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[5] UDVD：适用于可变降质类型的通用图像超分，附参考代码</a><br><a href="https://mp.weixin.qq.com/s/nj_C_LXFpWQZASc4ITJmmA" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[6] NTIRE2020冠军方案RFB-ESRGAN：带感受野模块的超分网络</a><br><a href="https://mp.weixin.qq.com/s/laZ_LH8tQ4PQaQXIyr38Bg" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[7] 超越RCAN，图像超分又一峰：RFANet</a><br><a href="https://zhuanlan.zhihu.com/p/147171955" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[8] #每日五分钟一读#Image Super-Resolution</a><br><a href="https://zhuanlan.zhihu.com/p/148965379" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[9] CVPR 2020 | 几篇GAN在low-level vision中的应用论文</a><br><a href="https://mp.weixin.qq.com/s?__biz=MzU5MTgzNzE0MA==&amp;mid=2247486455&amp;idx=1&amp;sn=94156f53845cdf07e859e343862be745&amp;chksm=fe29a94cc95e205ae880cb9518cb16a54c0abde69e5a8654f7fedf70a0cc990775265bf28a1c&amp;mpshare=1&amp;scene=23&amp;srcid=&amp;sharer_sharetime=1592478384281&amp;sharer_shareid=e2c873c63ba339b9a32a1bfa460a6ebf#rd" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[10] 超100篇！CVPR 2020最全GAN论文梳理汇总！</a><br><a href="https://mp.weixin.qq.com/s?__biz=MzU5MTgzNzE0MA==&amp;mid=2247484246&amp;idx=1&amp;sn=3af777e066a3d4b6dcf1c4cf4856a671&amp;chksm=fe29a1edc95e28fb85299ccf87a1819755f80243ff17f0057c425f3c4077e7e3b4d4a2035329&amp;scene=21#wechat_redirect" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[11] CVPR2020之MSG-GAN：简单有效的SOTA</a><br><a href="https://github.com/amusi/CVPR2020-Code" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[12] CVPR2020-Code</a><br><a href="https://zhuanlan.zhihu.com/p/117341018" style="font-size: inherit; line-height: inherit; margin: 0px; padding: 0px; text-decoration: none; color: rgb(30, 107, 184); overflow-wrap: break-word;">[13] 慢镜头变焦：视频超分辨率：CVPR2020论文解析</a></p></div>
