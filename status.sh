cd ~/DLNLP
RUN_ID=$(ls -t experiments/report_pipeline/ | head -1)
BASE="experiments/report_pipeline/$RUN_ID"
echo "Run ID: $RUN_ID"
echo "RQ1:      $(find $BASE/rq1 -name results.json 2>/dev/null | wc -l)/9"
echo "RQ2:      $(find $BASE/rq2 -name results.json 2>/dev/null | wc -l)/15"
echo "Ablation: $(find $BASE/ablation -name results.json 2>/dev/null | wc -l)/27"
echo ""
echo "--- Process ---"
ps aux | grep run_full_pipeline | grep -v grep
echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader