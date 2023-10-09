# FAQ

* **Q: The source and permissions to release data. The utility and scope of the dataset, how it might be employed by future researchers and benchmarks against existing methods?**

    We commit to releasing 5,195 of the 8,448 CT volumes upon acceptance. We have elaborated on the source and permissions in Table 3 ([supplementary](https://www.cs.jhu.edu/~alanlab/Pubs23/qu2023abdomenatlas.pdf)). To clarify, we will only disseminate the annotations of the CT volumes separately, and users will have to retrieve the original CT volumes, if needed, from the original sources (websites). Everything we intend to create and license-out will be in separate files and no modifications are necessary to the original CT volumes. We have consulted this with the lawyers at Johns Hopkins University, confirming the permissions of distributing the annotations based on the license of each dataset.

    1. **Benchmarking existing segmentation models.** We benchmarked the most recent advances in medical segmentation, i.e., SwinUNETR [[Tang et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Self-Supervised_Pre-Training_of_Swin_Transformers_for_3D_Medical_Image_Analysis_CVPR_2022_paper.pdf)], U-Net, UNETR and SegResNet.

    2. **A large dataset for developing medical foundation models.** Developing foundation models for healthcare has recently raised much attention. Foundation model refers to an AI model that is trained on a large dataset and can be adapted to many specific downstream applications. This requires a large-scale, fully-annotated dataset. We anticipate that our AbdomenAtlas-8K can play an important role in achieving this by enabling the model to capture complex organ patterns, variations, and features across different imaging phases, modalities, and a wide range of populations. This has been partially evidenced by our recent publication [[Liu et al., ICCV 2023](https://arxiv.org/abs/2301.00785)] and an ongoing project, showing that fully-supervised pre-training on AbdomenAtlas-8K transfers much better than existing self-supervised pre-training. 
    
    3. **The proposed strategy can scale up annotations quickly.** This strategy can be used for creating many medical datasets (across organs, diseases, and imaging modalities) or even natural imaging datasets. We are implementing our active learning procedure to re-create the natural imaging datasets previously produced by us [[He et al., ECCV 2022
    ](https://arxiv.org/abs/2112.00933); [Zhao et al., ECCV 2022](https://arxiv.org/abs/2111.14341)], yielding a considerably reduced amount of labeling efforts. Moreover, our strategy is being integrated into open-source software such as MONAI-LABEL at NVIDIA and ChimeraX at UCB/UCSF. This will make a difference in the rapid annotation of medical images in the near future.
    
    4. **Enabling precision medicine for various downstream applications.**
    We showcased one of the most pressing applications—early detection and localization of pancreatic cancer, an extremely deadly disease, with a 5-year relative survival rate of only 12% in the United States. The AI trained on a large, private dataset at Johns Hopkins Hospital (JHH), performed arguably higher than typical radiologists [[Xia et al., medRxiv 2022](https://www.medrxiv.org/content/10.1101/2022.09.24.22280071v1)]. But this AI model and annotated dataset were inaccessible due to the many policies. Now, our paper demonstrated that using AbdomenAtlas-8K (100% made up of publicly accessible CT volumes), AI can achieve similar performance when directly tested on the JHH dataset (see Table 2). This study is a concrete demonstration of how AbdomenAtlas-8K can be used to train AI models that can be generalized to many CT volumes from novel hospitals and be adapted to address a range of clinical problems.

    [Follow-up Plans] Upon the dataset's release, we aim to host international competitions in 2024, addressing the challenges in multi-organ and multi-tumor segmentation at scale via platforms such as MICCAI/RSNA/Grand Challenge.
    Two contributions: A large-scale dataset of 8,448 annotated CT volumes and an active learning procedure that can quickly create many other large-scale datasets.

* **Q: Comparing AbdomenAtlas-8K with AMOS, AbdomenCT-1K, TotalSegmentator**

    1. **A significantly larger number of annotated CT volumes.** TotalSegmentator, AMOS, and AB1K provided 1,204, 500, and 1,112 annotated CT volumes. AbdomenAtlas-8K provided 8,448 annotated CT volumes (around eight times larger).
    
    2. **A notably greater diversity of the provided CT volumes.** The CT volumes in AbdomenAtlas-8K were collected and assembled from at least 26 different hospitals worldwide, whereas the makeup of TotalSegmentator and AMOS was sourced from a single country. Specifically, TotalSegmentator was from Switzerland (biased to the Central European population) and AMOS was from China (biased to the East Asian population). While AbdomenCT-1K was from 12 different hospitals, our AbdomenAtlas-8K presents significantly more CT volumes (8,448 vs. 1,112) and more types of annotated classes (8 vs. 4). 
    
    3. **The manual annotation time was significantly reduced.** The creation of AbdomenAtlas-8K used an effective active learning procedure, reducing the annotation time from 30.8 years to three weeks (see the revised Section 3.3 for a detailed calculation). This is an important scientific attempt to put active learning into practical use.
    
    4. **Produce an attention map to highlight the regions to be revised.** The attention maps mentioned in our active learning procedure can accurately detect regions with a high risk of prediction errors (evidenced in Table 1). This capability enables annotators to quickly find areas that require human revision. As a result, it significantly reduces annotators' workload and annotation time by a factor of 533.

    The  following Table to signify the extension of this research over the previous ones.
    |  dataset name  | # of CT volumes  | # of annotated organs | # of hospitals | use of active learning | license |
    |  ----  | ----  |  ----  | ----  | ----  | ----  |
    | AMOS | 500 | 15 | 2 | No | CC BY 4.0 | 
    | AbdomenCT-1K | 1,112 | 4 | 12 | No | CC BY 4.0 |
    | TotalSegmentator | 1,204 | 104 | 1 | No | CC BY 4.0 |
    | AbdomenAtlas-8K | 8,448 | 8 | 26 | Yes | pending |

    **[Comparing with TotalSegmentator]** In TotalSegmentator, the assertion, *“all 1,204 CT examinations had annotations that were manually reviewed and corrected whenever necessary”* (quote from the original paper) stated two crucial points. Therefore, (1) the labels were largely generated by a **single** nnU-Net re-trained continually and (2) the annotators must review and revise **every** CT volume whenever necessary. Thereby, the main difference between TotalSegmentator and AbdomenAtlas-8K is two-fold:
    1. **Reducing architectural biases.** The labels in AbdomenAtlas-8K were averaged from **three** segmentation architectures. Depending solely on nnU-Net could introduce a potential label bias favoring the nnU-Net architecture. This means that whenever TotalSegmentator is employed for benchmarking, nnU-Net would always outperform other segmentation architectures (e.g., UNETR, TransUNet, SwinUNet, etc.). This observation has also been made in several publications that used the TotalSegmentator dataset. For example, Table 3 in [[Huang et al., 2023](https://arxiv.org/abs/2304.06716)] showed that nnFormer, UNETR, and SwinUNETR were all outperformed by nnU-Net in TotalSegmentator (of course their proposed architecture surpassed nnU-Net, but this required independent validation). Hence, the use of an ensemble of three (or even more) models was crucial to reduce such an architectural bias.
    2. **Reducing annotation time.** As outlined in the TotalSegmentator paper, the annotators must review and revise (when necessary) every CT volume (1204/1204 volumes) without priority. In contrast, we have introduced an innovative active learning procedure that identifies the most crucial CT volumes and significant regions within them, guiding annotators to put their efforts more accurately. Consequently, annotators are now only tasked with revising **400/8448** volumes. This represents a substantial reduction (from 30.8 years to three weeks) in the radiologists' workload through algorithmic innovation.

    Concurrently, we are actively expanding the classes covered by the AbdomenAtlas-8K. Starting with the set of 104 classes found in TotalSegmentator, we aim to further diversify the range of classes covered. We believe this progressive enhancement will further bridge the gap and accentuate the complementary nature of our dataset to TotalSegmentator and other public datasets.

* **Q: How do your proposed labels more accurate information over existing labels.**

    It is true that we did not introduce new CT volumes. After summarizing the existing public datasets (see Table 3), we found that a combination of these datasets is fairly large (a total of 8,000 CT volumes and many more are coming over the years). The main challenge, however, is the absence of comprehensive annotations. In fact, scaling up annotations is much harder than scaling up CT volumes due to the limited time of expert radiologists. Our contribution is, therefore, significant, covering two major perspectives.
    
    1. **More comprehensive annotations.** We provide per-pixel annotations for eight organs across over 8000 CT volumes, encompassing various phases and diverse populations. The existing labels of the public datasets are partial and incomplete, e.g., LiTS only had labels for the liver and liver tumors, and KiTS only had labels for the kidneys and kidney tumors. Our AbdomenAtlas-8K offered detailed annotations for all eight organs within each CT volume. As praised by reviewer nhxF: *“As currently the largest dataset, the work significantly contributed to the field of abdominal CT imaging and can serve as the basis of the development of effective AI algorithms in the future.”*
    
    2. **A fast procedure to annotate CT volumes at scale.** Our active learning procedure enables us to accomplish the task within three weeks, a remarkable contrast to the conventional manual annotation approach, which typically requires about 30.8 years. This accelerates the annotation process by an impressive factor of 533. The majority of the annotation workload is managed solely by only one radiologist. Furthermore, this strategy also allows us to efficiently annotate a wider range of organs and tumors in the future.
    
    3. [Minor] **Novel dataset.** We have applied our active learning procedure to creating a large private dataset (JHH) of 5,281 novel CT volumes with per-voxel annotations. However, at the moment, we cannot guarantee that this dataset can be released soon due to numerous regulatory constraints. We are actively working with Johns Hopkins Hospital to expedite its potential release. This has been clarified in our introduction: “we commit to releasing 5,195 of the 8,448 CT volumes.”

* **Q: The dataset of 8000 annotated CT images created using the annotation tool is not available yet.**

    We used commercial software called [Pair](https://aipair.com.cn/) to perform annotation reviews and revisions, as specified in footnote #4. Acquiring this software requires purchasing a license for each individual user. Subsequently, we discovered that [MONAI-LABEL](https://monai.io/label.html) is also a very useful tool for the annotation task. 

* **Q: What is the JHH dataset and "pseudo labels"**

    Note that the JHH dataset is a proprietary, multi-resolution, multi-phase collected from Johns Hopkins Hospital. Both arterial and venous phase scans were collected for each subject, so there were 5,282 CT volumes (2,641 subjects) in total.

    Pseudo labels refer to organ labels predicted by AI models without any additional revision or validation by human annotators.

* **Q: In the **revised** Table 5 (supplementary), the improvement on JHH, specifically in terms of mDice and mNSD, is around 1-2 points. This falls within the standard deviation for each of the 9 organs on average.**

    Table 5 (supplementary) only showed the improvement from Step 0 to Step 1 of the active learning procedure, instead of presenting the full improvement before and after using our active learning procedure.

    1. We added the improvement of Step 2 in the active learning procedure. Compared to the results in step 0, both mDSC and mNSD show substantial improvement, confirming the effectiveness of our active learning procedure. 
    2. We marked the revised classes at each Step. The segmentation performance of some organs were marginally improved simply because (1) these organs were hardly revised at certain Steps or (2) the segmentation performance was already very high (e.g., DSC > 90%). For those revised organs (e.g., aorta), AI models showed a significant improvement from 72.3% to 82.9%. 

* **Q: In the **revised** Table 4 (supplementary), there's a record of the annotation time for each of the three annotators. The abstract notes a significant reduction in annotation time from 30.8 years to three weeks. It would be beneficial to have a clearer breakdown of the time savings per scan using the proposed method versus the traditional approach. Additionally, some insights into the offline time required for model training and testing would be valuable.**

    1. **Why 30.8 years?** We considered an 8-hour workday, five days a week. A trained annotator typically needs 60 minutes per organ per CT volume [[Park et al., 2020](https://www.sciencedirect.com/science/article/pii/S2211568419301391)]. Our AbdomenAtlas-8K has a total of eight organs and around 8,000 CT volumes. Therefore, annotating the entire dataset requires 60x8x8000 (minutes) / 60/8/5 = 1600 (weeks) = 30.8 (years).
    2. **Why three weeks?** Using our active learning procedure, only 400 CT volumes require manual revision from the human annotator (15 minutes per volume). That is, we managed to accelerate the annotation process by a factor of 60x8/15=32 per CT volume. Therefore, we completed the entire annotation procedure within three weeks as reported in the paper. Human efforts: 400x15 (minutes) / 60/8= 12.5 (days) plus the time commitment for training and testing AI models taking approximately 8.5 (days).

* **Q: In Table 1, the authors compare attention masks to ground truth using sensitivity and precision. It'd be helpful to know why other metrics like specificity, recall, mDice, and mNSD, mentioned elsewhere in the submission, weren't included here.**

    1. **Why sensitivity and precision**. We chose sensitivity and precision because Table 1 strived to evaluate an error detection task rather than a segmentation task (comparing the boundary of attention maps and error regions). Once an error is detected, it counts as a hit, otherwise, as a miss. Sensitivity and precision can measure how well the attention maps detect the real error regions and whether the errors being detected are real errors, respectively. Hence, these two metrics are appropriate. 
    2. **Why not specificity, recall, mDice, or mNSD?** Recall and sensitivity are the same by [definition](https://en.wikipedia.org/wiki/Sensitivity_and_specificity). Given that the non-error region (true negatives) significantly outweighs the error region, specificity approaches a value close to one (see the table below). Thereby, specificity cannot give us meaningful information on the error detection performance. Moreover, mDice and mNSD are commonly used to evaluate the similarity of two masks, but as discussed above, the similarity in the boundary between attention maps and error regions is not necessary. Following your suggestion, we computed mDice, mNSD, and specificity in the table below. You can see that they do not provide meaningful information on the error detection performance of attention maps.

    |  organ  | mDice(%)  | mNSD(%) | Specificity |
    |  ----  | ----  |  ----  | ----  | 
    | Spleen | 1.0 | 1.5 | 0.99  |
    | Right Kidney | 10.1 | 11.8 | 0.99  |
    | Left Kidney | 12.3 | 15.2 | 0.99 |
    | Gallbladder | 11.7 | 17.5 | 0.99 |
    | Liver | 19.1 | 21.1 | 0.99  |
    | Stomach | 26.3 | 29.3 | 0.99 |
    | Aorta | 27.4 | 20.4 | 0.99 |
    | Postcava (IVC) | 18.8 | 20.8 | 0.99 |
    | Pancreas | 18.5 | 24.0 | 0.99 |

* **Q: Since the algorithm itself determines the priority list, there's no assurance that the remaining 92.5% are accurate. There could be instances where the attention map score might not predict an error even when there's a significant abnormality that all three DL models overlook.**

     It is true that the remaining 92.5% have a potentially important abnormality in all three models. Our solutions are in two dimensions:
    1. **Quickly review the entire dataset**. Two junior radiologists (3-year experience) were responsible for looking through the entire AbdomenAtlas-8K once the active learning procedure was completed. The radiologists would make a revision if the labels were incorrect, but such revisions were seldom required, with only 55 out of 8,448 instances needing adjustments. This strategy guaranteed the automated AI annotation quality in the remaining 92.5%.
    2. **Enrich the AI architectures used for dataset construction**. We plan to unify segmentations predicted by more AI architectures. This strategy can significantly attenuate the prediction errors made by specific AI architectures in the remaining 92.5%.

* **Q: In Appendix A, training with the new dataset doesn't show a significant increase in Dice (90.3% vs. 90.4%). It might benefit from statistical tests to better highlight the utility of the expanded dataset.**

    The marginal improvement in the FLARE’23 dataset can be due to many reasons. For example, the performance of the eight specific organs is already very high (DSC > 90%) and training with more annotations may not yield significant returns. 
    Moreover, we also recognized that limiting our evaluation exclusively to the FLARE'23 dataset may not provide a comprehensive assessment, as its small sample size (*N* = 300) may not present the full spectrum of different domains. To improve, we have evaluated the trained models on two additional (unseen) datasets with larger sample sizes (*N*): TotalSegmentator (*N* = 1,204) and JHH (*N* = 5,281). Following your suggestion, we have conducted a statistical analysis of the comparison. The mean, standard deviation, and *p*-value are reported in the revised Table 6. We obtained a more noticeable benefit from a larger scale evaluation when training AI models on AbdomenAtlas-8K over the previously partially labeled ones.

* **Q: It might be helpful to look at the number of revisions made by each organ and share those statistics. This could shed light on the varying complexities of segmenting different organs.**

    We have reported the amount of revisions by highlighting the two most significantly revised classes (i.e., aorta and postcava) at each step in blue in Appendix Table 5. These two organs have consistently shown a steady rise in mDSC during the active learning procedure. Specifically, the aorta increased from 72.3% to 83.7%, and the postcava improved from 76.1% to 78.6%.

    In addition, some organs are harder to segment than others due to two reasons.

    1. **Diverse annotation protocols.** As mentioned in A2, there are often varying annotation protocols for the aorta among different hospitals. Consequently, achieving precise segmentation with our AI models becomes a challenging task. This often requires frequent revisions of aorta annotations by our annotators. The improvement of aorta annotations through the active learning procedure is illustrated in Table 5. Notably, there is a substantial increase in mDSC, from 72.3% to 83.7%, after undergoing two steps of active learning procedure.
    2. **Blurry organ boundary.** Organs such as the pancreas often exhibit blurry boundaries, presenting a challenge for both AI and human annotations. Consequently, there is no observed improvement in annotation performance for such organs even after two steps of our active learning procedure.

* **Q: In Figure 3, the sum of attention maps for each organ is depicted. Clarifying the experiment's intent and the reasoning behind the 5% threshold would be beneficial.**

    1. **The intention behind Figure 3** is to visualize the distribution of the attention size of each CT volume. A larger attention size implies a greater requirement for revision in various regions.  Figure 3 suggests that the attention size for most CT volumes is small, but there are several significant-sized outliers. These outliers are of high priority for revision by human experts. According to the figure, the ratio of outliers is about 5% (highlighted in red). The 5% is estimated by the plot and also related to the budget of human revision for each Step in the active learning procedure. It is essential to emphasize that roughly 5% of CT volumes within each dataset are highly likely to contain predicted errors, requiring further revision by our annotator. 
    2. **The choice of the 5% threshold.** It is true that this number is empirical. The 5% is estimated based on (1) empirical observations (the number of outliers in Figure 3) and (2) the annotation budget at each Step in the active learning procedure. If there are many outliers or a limited budget, the threshold needs to be increased accordingly. If one is dealing with new datasets, this threshold is easily obtained by analyzing the attention size distribution as plotted in Figure 3.

* **Q: Regarding Figure 5, the attention map appears to highlight regions beyond FP/FN labels or ambiguous cases. If these regions also underwent human review, it might lead to extra, potentially needless, effort.**

    Reducing such regions is one of the main focuses in the active learning literature. Compared with random selection and revision, our proposed method has yielded a significant reduction in human curation. But certainly, there are regions where AI made correct predictions but our method still asked humans to review and where AI made mistakes but our method cannot detect them. To analyze and overcome this problem, we took two actions in this study.
    1. We have quantified these regions in Table 1, suggesting that for some organs, our method presented high sensitivity and precision (both > 0.9), while some organs had relatively lower sensitivity (e.g., liver) and precision (e.g., spleen, gallbladder, and IVC). Consequently, the active learning methods for these organs require further investigation.
    2.  Two junior radiologists (3-year experience) took the responsibility to look through the entire AbdomenAtlas-8K once the active learning procedure is completed. The radiologists will make a revision if the labels were incorrect (i.e., our method missed the regions), but such revisions were seldom required, with only 55 out of 8,448 instances needing adjustments. 

* **Q: It would be helpful to understand the method used for the assignment of three human annotators and the training level or expertise of the annotators.**

    Our study recruited three annotators, comprising a senior radiologist with over 15-year experience and two junior radiologists with 3-year experience. The senior radiologist undertook the task of annotation revision in the active learning procedure. The junior radiologists were responsible for reviewing the completed AbdomenAtlas-8K annotations and making revisions if needed. Additionally, they conducted the inter-annotator variability analysis in Figure 8 and further recorded the time required for annotating each organ in Table 4.

    Figure 8(b) presented the inter-annotator agreement across the two human experts and one AI model. Overall, the agreement between the two human experts is fairly high (DSC > 0.9). Several organs have a relatively lower agreement, such as the gallbladder (DSC = 0.85) and pancreas (DSC = 0.81). It is because the boundaries of these two organs are often blurry and ambiguous.

    The purpose of investigating the inter-annotator agreement in Figure 8(b) is to assess the quality of AI predictions for organ segmentation. Therefore, in Figure 8(b), we asked the two radiologists to annotate these organs from scratch without the use of initial AI annotation—**the AI annotation will not bias the radiologists**. As a result, for each organ, we obtained two annotations made independently by the two human experts and one annotation predicted by the AI model. Figure 8(b) presented the mutual DSC score of each pair, showing that AI can segment these organs with a similar variance to human experts.

    One annotator is a trained radiologist with 15 years of experience. The other two additional annotators are with 3 years of experience.

* **Q: Does the active learning model improve performance of a single annotator, or all annotators. In other words, if the segmentations from all three annotators were combined (without the proposed strategy), would they be better or worse than segmentation acquired using the proposed approach?**

    Only one annotator was assigned to revise the annotations generated by AI models. We did not observe a substantial improvement in the annotator's performance in organ segmentation. This is because annotating organ boundaries is relatively less difficult for human experts with 15-year experience than annotating tumors. Therefore, our strategy has rather minor contributions to enhancing the annotation performance of human annotators. For organ segmentation, the pressing challenge lies in the time-consuming nature of the conventional full organ annotation process. In this regard, our strategy accelerates over 500 times compared with conventional ones.

* **Q: It would be beneficial to understand if annotation procedures vary across hospitals or among annotators, possibly due to differences in training or other factors, and how these variations might influence performance.**

    Consistency and accuracy are of utmost importance in medical data annotation. However, in the real world, variations in annotation procedures can occur across hospitals, laboratories, or individual annotators. When creating AbdomenAtlas-8K, we observed the variations in annotation procedures across 26 hospitals in our study. Two factors could cause these variations based on our observation.
    1. **Different scanning body range.** An illustrative instance is the aorta, an elongated anatomical structure traversing both the thoracic and abdominal cavities. Notably, its morphology exhibits significant differences between these distinct anatomical regions. For abdominal CT scans, in general, the aortic arch in the chest region is typically not scanned. However, due to variations in the scanning range of CT scans across different hospitals, some CT scans also include and annotate the aortic arch. As a result, the standard annotation for the same organ provided by different hospitals varies significantly. 
    2. **Different annotation protocols across hospitals.** For example, the stomach and duodenum often have blurry boundaries, posing a challenge in distinguishing between these two organs. Different hospitals typically adhere to varying annotation protocols, which can further complicate the learning process for our AI model.

    **Positive impact to AI training:**
    - Variability to some extent could enhance the robustness of AI models. If a model is trained on a diverse set of annotations from different sources, it might be better equipped to generalize to novel data.

    **Negative impact to AI training:**
    - AI models trained on inconsistent annotations could produce unreliable or unpredictable outcomes. As a result, it might be challenging to reproduce AI predictions across different hospitals if there's significant variability in the source annotations.

* **Q: Some further discussion on the feasibility of he generalisability of the approach and the work that would be involved to adapt the software would be useful.**

    We have undertaken several explorations to assess the generalisability of our approach, as outlined below.
    1. **Generalisability to other organs/tumors.** We continued to annotate 13 additional organs and vessels. (e.g., esophagus, portal vein and splenic vein, adrenal gland, duodenum, hepatic vessel, lung, colon, intestine, rectum, bladder, prostate, head of femur, and celiac trunk.), and different types of tumors (e.g., kidney tumor, liver tumor, pancreas tumor, hepatic vessel tumor, lung tumor, colon tumor, and kidney cyst). Leveraging our proposed approach can significantly accelerate the annotation of various anatomical structures. However, annotating more types of tumors remains challenging if solely using our approach; even human experts may face uncertainty in tumor annotation, particularly in the early stages of the disease. Consequently, further research is warranted. This could entail incorporating additional data sources such as radiology reports, biopsy results, and patient demographic information into the tumor annotation.
    2. **Generalisability to other imaging modalities.** In this work, we have applied our approach to a number of CT volumes across different phases, including portal (44%), arterial (37%), pre- (16%), and post-contrast (3%) phases. Given the inherent differences in these scans, we believe that our active learning procedure can be extended to datasets from other imaging modalities, including MRI, Ultrasound, Histopathology images, etc.
    3. **Adapt to the software.** Our active learning procedure is being integrated into open-source software such as [MONAI-LABEL](https://monai.io/label.html) at NVIDIA and [ChimeraX](https://www.cgl.ucsf.edu/chimerax/) at UCB/UCSF.

* **Q: One of the limitations of the approach is evaluating the proposed methodology with respect to disease (e.g., when tumors are present). It would also be important to highlight that the algorithm used in the active learning step is supervised, therefore, any label biases from labels in the training datasets may potentially be propagated into the labeling of the new data.**

    Indeed, as discussed in our limitation section, tumor annotation is much more challenging than organ annotation. It is true that when annotators make mistakes on tumor annotations, it will also propagate the AI training and predicting along the supervised active learning procedure. As an extension of this study, we will add tumor annotations to the AbdomenAtlas-8K  dataset by addressing the challenge in two possible directions. **Firstly**, we plan to recruit more experienced radiologists to revise tumor annotations. **Secondly**, we will incorporate the radiology reports (based on biopsy results) into the human revision. These actions can reduce potential label biases and label errors from human annotators.