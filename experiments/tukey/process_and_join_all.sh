julia process_discriminability_data.jl /home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_discriminability_contaminated-0.00_pre
julia process_discriminability_data.jl /home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_discriminability_contaminated-0.00_pre
julia process_discriminability_data.jl /home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_discriminability_contaminated-0.01_pre
julia process_discriminability_data.jl /home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_discriminability_contaminated-0.05_pre

julia join_discriminability_normal_data.jl /home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_discriminability_contaminated-0.00_post
julia join_discriminability_normal_data.jl /home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_discriminability_contaminated-0.00_post
julia join_discriminability_normal_data.jl /home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_discriminability_contaminated-0.01_post
julia join_discriminability_normal_data.jl /home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_discriminability_contaminated-0.05_post

julia ../eval_paper/rank_tables.jl --discriminability
julia ../eval_paper/measure_comparison_tables.jl --discriminability
julia ../eval_paper/correlation_tables.jl --discriminability
julia ../eval_paper/multiclass_comparison.jl --discriminability