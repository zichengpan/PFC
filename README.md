 ---

<div align="center">    
 
# Overcoming Learning Bias via Prototypical Feature Compensation for Source-Free Domain Adaptation

[Zicheng Pan](https://zichengpan.github.io/), Xiaohan Yu, Weichuan Zhang, and Yongsheng Gao

[![Paper](https://img.shields.io/badge/paper-Pattern%20Recognition-blue)](https://www.sciencedirect.com/science/article/pii/S0031320324007763)

</div>

## Abstract
The focus of Source-free Unsupervised Domain Adaptation (SFUDA) is to effectively transfer a well-trained model from the source domain to an unlabelled target domain. During the target domain adaptation, the source domain data is no longer accessible. Prevalent methodologies attempt to synchronize the data distributions between the source and target domains, utilizing pseudo-labels to impart categorical information, which has made some progress in improving the model's performance. However, performance impairments persist due to the introduction of learning bias from the source model and the impact of noisy pseudo-labels generated for the target domain. In this research, we reveal that the central cause for feature misalignment during domain transition is the learning bias, which is generated by the discrepancy of information between source and target domain data. The source domain data may contain distinguishable features that do not appear on the target domain, which causes the pre-trained source model to fail to work during domain adaptation. To overcome the information discrepancy, we propose a Prototypical Feature Compensation (PFC) Network. The network extracts representative feature maps of the source domain. Then use them to minimize the discrepancy information in the target domain feature maps. This mechanism facilitates feature alignment across different domains, allowing the model to generate more accurate categorical data through pseudo-labelling. The experimental results and ablation studies demonstrate exceptional performance on three SFUDA datasets and provide evidence of the proposed PFC method's ability to adjust the feature distribution of both source and target domain data, ensuring their overlap in the latent space. 

## Dataset Preparation Guideline
Please prepare the datasets according to the configuration files in the data folder. You may refer to these sites to download the datasets: [Office-31 / Office-Home](https://github.com/tim-learn/SHOT), [DomainNet-126](https://ai.bu.edu/M3SDA/). The default data structure is given as follows:

```
./data
|–– office/
|–– domainnet-126/
|–– office-home/
|   |–– domain1/
|   |–– domain2/
|   |-- ...
|   |–– domain1.txt
|   |-- domain2.txt
|   |-- ...
```

## Training Scripts
Training scripts are provided for each dataset separately (office31.sh, home.sh, domainnet.sh). To train the model, simply execute the appropriate script. For example, to run the experiment on DomainNet-126, use the following command:
    ```
    sh domainnet.sh
    ```
    
## Citation
If you find our code or paper useful, please give us a citation, thanks!
```
@article{pan2024overcoming,
    title = {Overcoming Learning Bias via Prototypical Feature Compensation for Source-free Domain Adaptation},
    author = {Zicheng Pan and Xiaohan Yu and Weichuan Zhang and Yongsheng Gao},
    journal = {Pattern Recognition},
    pages = {111025},
    year = {2024},
    publisher={Elsevier}
}
```

## Acknowledgment
We thank the following repos for providing helpful components/functions in our work.

- [SHOT](https://github.com/tim-learn/SHOT)

- [AaD](https://github.com/Albert0147/AaD_SFDA)

  
