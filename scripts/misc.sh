# tabulate labels in training data
for f in data/index/Xiphorhynchus_guttatus_vs_elegans/*; do echo $f; cut -d/ -f1 < $f | sort | uniq -c; done

# extract species labels
prettier ~/tmp/elaenia/xeno-canto/dataset/metadata/gen_Xiphorhynchus/page*.json | jq .recordings[].sp | sort | uniq
