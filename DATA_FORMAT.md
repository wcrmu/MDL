# MDL HDFS and Parquet Data Contract

Last consolidated: 2026-07-18

This document describes the upstream HDFS dataset, the two supported Parquet
layouts, and every physical or adapter-generated field consumed by the current
MDL data path. The runtime source of truth is
configs/mdl_rankmixer.yaml together with
src.dataloader:adapt_mdl_rankmixer_parquet.

The upstream Parquet schema contains 630 physical columns. A complete
630-column name dump is not stored in this repository, so this document does
not invent names for unused columns. It does enumerate the complete current
projection: all 169 non-sequence model fields, all 107 raw UPS attributes,
request metadata, labels, aggregation indices, optional prediction metadata,
and adapter-generated fields.

## 1. HDFS location and partition layout

Base URI:

~~~text
hdfs://temu-data-ns/apps/nothive/warehouse/searchrec/searchrec_cvr_allscene_agg_fgoutput_hour_dracarys_exp
~~~

The dataset uses a two-level Hive-style directory layout:

~~~text
base_dir/
├── pt=2026-07-10/
│   ├── hr=00/
│   │   ├── 000000_0.gz.parquet
│   │   ├── 000001_0.gz.parquet
│   │   └── ...                         approximately 500 files/hour
│   ├── hr=01/
│   ├── hr=02/
│   ├── hr=03/
│   └── hr=04/
├── pt=2026-07-09/
└── ...                                 21 date directories in the observed snapshot
~~~

The observed file-level characteristics are:

| Property | Observed value |
|---|---:|
| Container format | Parquet |
| Column compression | GZIP |
| File suffix | .gz.parquet |
| Files per hour | approximately 500 |
| Raw aggregate rows per file | approximately 1,728 |
| Raw aggregate rows per hour | approximately 864,000 |
| Physical columns | 630 |
| Physical names containing _hn | 427 |

The .gz.parquet suffix denotes a Parquet file using GZIP-compressed column
chunks; it is not treated as a plain text file wrapped in an external gzip
stream. PyArrow reads the codec from Parquet metadata.

The pt and hr values are partition directory keys. The training code does not
assume that they are also stored as physical columns inside every file.

The current mdl_rankmixer profile selects:

~~~text
train:
  pt=2026-07-10/hr=00
  pt=2026-07-10/hr=01
  pt=2026-07-10/hr=02

test:
  pt=2026-07-10/hr=04
~~~

Paths remain YAML configuration, not Python constants. The reader accepts a
file, directory, or glob, resolves hdfs:// and viewfs:// URIs through
PyArrow, recursively discovers files ending in .parquet, prunes columns, and
can shard work by Parquet row group across DDP ranks.

## 2. Reconciling the column counts

Several earlier counts measured different things and therefore must not be
compared as if they were all physical schema widths.

| Count | Meaning |
|---|---|
| 630 | All physical columns in the observed upstream Parquet schema |
| 427 | Physical schema names containing _hn in the upstream probe |
| 565 | Earlier reported YAML _hn entries/usages, including sequence and repeated use sites; not a unique physical-column count |
| 169 | Logical non-sequence fields used by the model: 47 request plus 122 candidate |
| 107 | Raw attributes across the nine UPS sequences: 9 timestamps plus 98 encoded categorical attributes |
| 281 | Mandatory raw columns projected by the current adapter |
| 12 | Optional train columns: agg indices plus one optional item feature |
| 13 | Optional test columns: the train optional set plus example_ids |
| 293 | Maximum unique train projection when all optional agg columns exist |
| 294 | Maximum unique test projection when all optional columns exist |
| 261 | Names in that maximum projection containing _hn; 250 end exactly in _hn and 11 end in _hn_share |

The mandatory count is 281 rather than 282 because
f_goods_view_times_tg_l1_hn is a logical model field but is declared as an
optional scan column. “Optional” here means the scanner projects it only when
the sampled Parquet schema contains it; it does not redefine the field as
semantically optional to the model.

Column pruning intentionally ignores the remaining upstream columns. They are
not part of the model input contract.

## 3. Supported physical row layouts

### 3.1 Aggregate layout: agg

One physical Parquet row contains several requests and their candidates.

| Logical content | Physical organization |
|---|---|
| Request/context fields | Outer axis is request position |
| Candidate/item fields | Outer axis is candidate position |
| Labels | One value per candidate |
| context_indices | Request identifier for every request/context position |
| target_indices | Request identifier for every candidate position |
| {ups}_x_indices | For every UPS event, the request identifiers allowed to see that event |

Example:

~~~text
context_indices = [10, 20]
target_indices  = [10, 10, 20]

request 10 -> context position 0, candidates 0 and 1
request 20 -> context position 1, candidate 2

impr_x_indices = [[10], [10, 20], [20]]
event 0 is visible to request 10
event 1 is visible to both requests
event 2 is visible to request 20
~~~

For each agg row:

- context_indices length is the request count;
- every request identifier in context_indices maps to exactly one context
  position;
- target_indices length is the candidate count;
- every request referenced by target_indices must exist in context_indices;
- each candidate field and each label has candidate_count entries;
- every {ups}_x_indices list is aligned with the corresponding raw UPS event
  arrays.

The adapter expands agg rows into candidate rows while caching normalized
request payloads once per request.

### 3.2 Request layout: req

One physical Parquet row represents one request and all of its candidates.
context_indices, target_indices, and all {ups}_x_indices columns are absent.
Request/context and UPS fields already belong to that one request; candidate
fields and labels retain a candidate outer axis.

Layout detection is structural:

- both context_indices and target_indices present means agg;
- both absent means req;
- only one present is an invalid mixed layout.

The old relocated data.md describes a different CTR dataset. Only its agg/req,
membership-index, and nested-array concepts apply here. Its field names, label
names, and UPS count are not part of this CVR contract.

## 4. Arrow nesting and flattening rules

The same Arrow type can represent different logical axes. Field membership in
the configured request, candidate, bag, and UPS lists is authoritative; Arrow
nesting alone is not.

| Logical value | Common agg representation | Common req representation |
|---|---|---|
| Request scalar category | list of singleton lists, one per request | scalar or singleton list |
| Request bag category | list of variable-length lists, one per request | one list; some producer variants retain an extra singleton axis |
| Candidate scalar category | list of singleton lists or list of scalars, one per candidate | same candidate axis |
| Candidate bag category | list of variable-length lists, one per candidate | same candidate axis |
| UPS scalar attribute | list of scalars or list of singleton lists, one per event | same sequence axis |
| context_indices / target_indices | list of int64 request identifiers | absent |
| {ups}_x_indices | list of lists of request identifiers | absent |

For low-level schema inspection:

- list<int64> needs one pc.list_flatten call to expose its values;
- list<list<int64>> needs two flatten operations to expose primitive values.

That is not a license to flatten every nested column mechanically. A true bag
or an event membership list uses the inner list as real data. Only known
singleton axes may be collapsed. In particular:

- non-sequence bag fields must preserve all tokens;
- the nine aligned SKU arrays must preserve token positions;
- {ups}_x_indices must preserve event-to-request membership;
- UPS attribute inner lists may be flattened only when each event token is a
  singleton.

The adapter normalizes each UPS attribute to list<int64>. The historical
test.parquet probe found all 107 UPS attributes stored as
list<list<int64>>, with an inner length of exactly one for every token.

### Null semantics

- A top-level null **or empty `[]`** UPS / optional bag / request-context list
  represents a zero-length value. Adapter canonical form is `[]`.
- Structure axes (`context_indices`, `target_indices`, candidate item/label lists)
  are **not** opened to empty-as-missing; they must keep their structural lengths.
- `{ups}_x_indices` is token-major: outer `[]` means zero UPS tokens (legal). An
  individual membership of `[]` on a present token is an orphan and is rejected.
- A null inside an UPS event on the configured `null_anchor_field` drops that
  whole step from every aligned field. Non-anchor nulls stay as padding ID 0 /
  dense 0.0.
- Sequence payloads expose `has_sequence = lengths > 0`. Empty `mean_pool`
  summaries are zero vectors; empty LONGER sequences keep learned CLS tokens.
- Dense scalar features append a presence bit: `null → value 0 + presence 0`,
  real `0 → presence 1`. Categorical null still maps to padding ID 0 only.
- A null inside an ordinary bag is masked.
- A null attribute at a valid SKU position keeps the position and pads only
  that attribute.
- Core candidate identifiers and all three labels are expected to be present.
  Complete-label contracts reject null / non-{0,1} on every flat batch.
- Candidates are not dropped merely because an optional feature is null.
- `trusted_input: true` may skip payload diagnostics after a one-row warm-up,
  but structure checks (UPS values/indices length alignment, membership,
  candidate/request outer lengths) stay on for every row.

## 5. Metadata, labels, indices, and generated columns

### 5.1 Request metadata

| Field | Scope | Meaning / use |
|---|---|---|
| search_id | request | Stable request/group identifier |
| scene_id | request | Raw scene value used for routing and per-scene AUC |
| impr_time | request | Request timestamp in milliseconds |

scene_id is distinct from scene_id_hn. The former is a small raw evaluation
dimension; the latter is an opaque encoded categorical model feature.

The current scenario configuration discovers sorted raw scene_id values from
the training split, accepts at most 64 values, and caches the mapping at
artifacts/scenarios/cvr_allscene.json.

### 5.2 Candidate metadata

| Field | Presence | Meaning / use |
|---|---|---|
| example_ids | Optional, principally test/prediction | Upstream candidate example identifier |
| candidate_position | Adapter-generated | Zero-based candidate position within a request |

Current prediction identity keys are search_id, candidate_position,
example_ids, and goods_id_hn.

### 5.3 Labels

| Task name | Raw label column | Contract |
|---|---|---|
| fst_cart | label_fst_cart | Complete binary 0/1 candidate label |
| upid_pay | upid_fst_trgt_noc_clk_pay_24h | Complete binary 0/1 candidate label |
| cateid_filter | cateid_is_fst_scene_sp_filter | Complete binary 0/1 candidate label |

cateid_is_fst_scene_sp_filter is a prediction target, not a row filter.
In agg files each label is a candidate-aligned array. The adapter emits scalar
int64 labels per candidate, and tensorization converts the three labels to the
training float tensor. The current production contract has no missing labels,
no label-mask columns, and no allocated label-mask tensor.

### 5.4 Aggregate-only membership columns

~~~text
context_indices
target_indices
impr_x_indices
clk_long_x_indices
view_long_x_indices
cart_long_x_indices
buy_long_x_indices
semi_clk_x_indices
srch_q2i_x_indices
ups_clk_sku_x_indices
flatten_query_hash_x_indices
~~~

## 6. Encoded categorical values

All fields containing _hn are upstream pre-encoded signed int64 bit patterns.
They are opaque categorical identifiers, not continuous quantities. Negative
values are normal: an earlier 427-column probe classified 299 columns as
all-negative, 89 as all-positive, and 39 as containing both signs.

The model reserves index 0 for null/padding. For a power-of-two bucket count,
each non-null signed int64 value is mapped by its low bits:

~~~text
embedding_id = (value & (bucket_size - 1)) + 1
valid real IDs: 1 ... bucket_size
embedding table rows: bucket_size + 1
~~~

Do not apply abs(), a second string hash, a sample-min offset, or a per-field
salt. Train, evaluation, and prediction must use the same bucket definition.

Six fields lack an _hn substring but follow the same pre-encoded categorical
contract:

~~~text
auto_price_p05_dis
auto_sales_p10_dis
mid_goods_prc_list_dis
mid_cmprc_diff_list_dis
ups_clkv2_i2i_goods_ids_hit_size
ups_clkv2_i2i_goods_ids_hit_all_size
~~~

The only raw UPS values with continuous time meaning are the nine
{ups}_x_time columns.

### Bag and aligned-SKU handling

The 50 fields marked B in the inventory are unordered categorical bags and
use masked mean pooling. The other 119 non-sequence fields are logical scalar
categories.

The following nine candidate bags are position-aligned SKU attributes:

~~~text
sku_id_hn
sku_price_v2_hn
sku_sales_hn
sku_spec_hash_hn
sku_spec_hn
sku_cart_cnt_7d_hn
sku_ordr_cnt_1m_hn
sku_price_dis_hn
sku_sales_dis_hn
~~~

Observed SKU-array lengths were 1 through 202. sku_spec_vids_hn belongs to the
broader SKU business group but was observed as a logical scalar and is not in
the nine-column aligned pooling group.

## 7. UPS sequence contract

All nine raw sequences are newest-to-oldest. The adapter first applies agg
request membership, then keeps the head of the filtered sequence, so only the
most recent configured number of events survives. Time deltas and Arrow output
are created only for retained events.

For retained event time t and request time impr_time:

~~~text
delta_seconds = (impr_time - t) / 1000
time_delta_log1p_seconds = log1p(delta_seconds)
~~~

The categorical {ups}_x_timegap_hn field is a coarse encoded bucket and is not
a replacement for the derived continuous time delta. Where causal processing
requires chronological order, the model reverses only the valid retained
window to oldest-to-newest; the adapter does not reverse the raw sequence.

| UPS | Raw attributes | Current head cap | Historical null rate | Historical median length | Historical observed max |
|---|---:|---:|---:|---:|---:|
| impr | 12 | 1,024 | 1.1% | 946 | 1,000 |
| clk_long | 12 | 2,048 | 0.4% | 1,340 | 8,000 |
| view_long | 30 | 2,048 | 0.6% | 1,183 | 8,000 |
| cart_long | 12 | 512 | 4.5% | 210 | 7,999 |
| buy_long | 13 | 256 | 26.7% | 22 | 5,559 |
| semi_clk | 6 | 128 | 32.6% | 4 | 199 |
| srch_q2i | 9 | 100 | 3.6% | 71 | 100 |
| ups_clk_sku | 10 | 200 | 2.5% | 200 | 200 |
| flatten_query_hash | 3 | 512 | 6.7% | 78 | 1,000 |
| Total | 107 | — | — | — | — |

The null rates, medians, and maxima above come from the relocated historical
sample document. They are observations, not full-HDFS guarantees and not
runtime truncation settings.

## 8. Complete non-sequence field inventory

Legend:

- S: logical scalar categorical field;
- B: unordered categorical bag;
- O: optional physical input in the current projection.

Every field below becomes an encoded categorical model input, including the
six fields without an _hn substring.

### 8.1 Context fields: 24 request-level fields

~~~text
S  currency_hn
S  hash_language_site_hn
S  language_hn
S  page_elsn_hn
S  page_sn_hn
S  plat_hn
S  region_hn
S  scene_id_hn
S  site_id_hn
S  timezone_hn
B  origin_query_hash_hn
B  query_arr_hn
B  query_hash_hn
B  query_terms_hash_hn
B  query_tfidf_term_hash_list_hn
B  query_extend_translation_hash_hn
S  search_method_hn
B  sess_q2q_hash_list_hn
B  recall_merge_cate_levels_hn
B  recall_merge_cate1_ids_hn
B  recall_merge_cate_ids_hn
S  scene_clk_cnt_15d_hit_hn
B  scene_impr_cnt_15d_hn
S  scene_impr_cnt_15d_hit_hn
~~~

### 8.2 User fields: 23 request-level fields

~~~text
S  uid_or_bg_hn
B  u_fst_ordr_cnt_mix_d_hn
B  clk_cnt_1d_hn
B  clk_3d_cnt_hn
B  clk_1d_cat_cnt_hn
B  cart_cnt_1d_hn
B  cart_cnt_3d_hn
B  clk_7d_page_sns_hn
B  clk_7d_page_elsns_hn
B  cart_7d_cat1_ids_hn
B  flip_mall_ids_hn
B  list_clk_cat1_ids_hn
B  list_clk_cat_ids_hn
B  ups_in_cart_2h_sku_cur_prices_hn
B  ups_in_cart_goods_hn_share
B  ups_incart_cat1_id_nc_hn
B  ups_in_cart_tg_hn
B  ups_query_term_hash_v2_hn
B  ups_query_tg_hn
B  ups_search_method_hash_hn
B  view_30m_cat1_ids_hn
B  view_7d_page_sns_hn
B  view_7d_page_elsns_hn
~~~

The 47 Context plus User fields contain 14 logical scalars and 33 bags.

### 8.3 Item fields: 104 candidate-level fields

~~~text
S    cat_id_hn
S    cat1_id_hn
S    cat2_id_hn
S    cat3_id_hn
S    cat4_id_hn
S    goods_id_hn
B    goods_name_bigram_hn
B    goods_ner_infos_hn
S    goods_scene_clk_cnt_15d_hn
B    goods_title_tfidf_term_hash_list_hn
S    goods_avlb_sku_num_dis_hn
S    goods_onsale_sku_num_dis_hn
S    goods_cluster_id_1w_hn
B    rev_ratings_cnt_crs_pos_hn
B    g_sku_spec_hn
B    g_sku_spec_hash_hn
B    g_sku_spec_unit_list_hn
B    g_prpty_val_id_list_hn
B    sku_id_hn
B    sku_price_v2_hn
B    sku_sales_hn
B    sku_spec_hash_hn
B    sku_spec_hn
S    sku_spec_vids_hn
B    sku_cart_cnt_7d_hn
B    sku_ordr_cnt_1m_hn
B    sku_price_dis_hn
B    sku_sales_dis_hn
S    price_hn
S    price_bef_coupon_hn
S    price_after_promotion_hn
S    price_after_promotion_div_hn
S    mkt_prc_hn
S    show_price_hn
S    is_promotion_hn
S    promotion_discount_hn
S    auto_price_p05_dis
S    auto_price_p10_dis_hn
S    ori_price_hn_share
S    sales_hn
S    auto_sales_p10_dis
S    c_adj_cart_cvr_15d_hn
S    c_adj_ctr_15d_hn
S    c_adj_ordr_cvr_15d_hn
S    c_cart_cnt_15d_hn
S    c_clk_cnt_15d_hn
S    c_impr_cnt_15d_hn
S    c_ordr_cnt_15d_hn
S    c_simi_adj_cart_cvr_15d_hn
S    c_simi_adj_ctr_15d_hn
S    c_simi_cart_cnt_15d_hn
S    c_simi_clk_cnt_15d_hn
S    c_simi_impr_cnt_15d_hn
S    idx_c_adj_cart_cvr_15d_hn
S    idx_c_adj_ctr_15d_hn
S    idx_c_adj_ordr_cvr_15d_hn
S    idx_c_cart_cnt_15d_hn
S    idx_c_clk_cnt_15d_hn
S    idx_c_impr_cnt_15d_hn
S    idx_c_ordr_cnt_15d_hn
S    idx_c_simi_adj_cart_cvr_15d_hn
S    idx_c_simi_adj_ctr_15d_hn
S    idx_c_simi_cart_cnt_15d_hn
S    idx_c_simi_clk_cnt_15d_hn
S    idx_c_simi_impr_cnt_15d_hn
S    scene_adj_cartcvr_15d_hn
S    scene_adj_ctr_15d_hn
S    scene_adj_cvr_15d_hn
S    scene_cart_cnt_15d_hn
S    nfk_sales_14d_hn
S    nfk_price_14d_hn
S    nfk_gmv_14d_hn
S    i2i2cat2_swing_hn
S    i2i_coclk_hn_share
S    i2i_list_amazoni2ifullgmv_hn_share
S    i2i_list_multimodal_hn_share
S    i2i_list_swingv3gmv_hn_share
S    i2i_hit_site_q2i_idx_hn
S    only_semi_swingi2i_cut60_hn_share
S    semi_swingi2i_cut30_hn_share
S    offline_outside_goods_id_list_hn_share
S    site_q2i_good_list_hn_share
S    multimodal_i2i_hit_cart_size_hn
S    multimodal_i2i_hit_clk_size_hn
S    main_goods_ids_hn_share
S    adj_cartcvr_hn
S    adj_ctr_hn
S    adj_cvr_hn
S    buy_long_spec_vids_hn
S    cart_long_spec_vids_hn
S    create_time_hn
S    mall_id_hn
S    sellr_type_hn
S    opt_id_hn
S    site_x_asian_code_hn
S/O  f_goods_view_times_tg_l1_hn
S    target_gs_last_cart_tg_hn
S    impr_3h_tg_hn
S    impr_all_tg_hn
S    impr_clk_6h_cnt_hn
S    clk_long_goods_abs_timegap_1d_hn
S    impr_long_goods_abs_timegap_1d_hn
S    mid_goods_prc_list_dis
S    mid_cmprc_diff_list_dis
~~~

### 8.4 Cross fields: 15 candidate-level fields

~~~text
S  rel_score_hn
S  rel_level_hn
S  q_hit_good_correct_unigram_hn
S  q2c_cart_15d_hit_val_hn
S  tit_in_top_query_cnt_hn
S  goods_query_emb32v3_cos_hn
S  query_cat_hn
S  query_pay_cnt_15d_hn
S  clk_hit_i2i_idx_hn
S  cart_hit_i2i_idx_hn
S  cart_long_hit_samestyle_i2i_idx_hn
S  ups_clkv2_i2i_goods_ids_hit_size
S  ups_clkv2_i2i_goods_ids_hit_all_size
S  us_ctr_price_dis50_hn
S  impr_cat_clk_goods_ids_cnt_1d_hn
~~~

### 8.5 Creative fields: 3 candidate-level fields

~~~text
S  ad_id_bin_hn
S  campaign_id_hn
S  idx_goods_creative_id_hn
~~~

The 122 Item plus Cross plus Creative fields contain 105 logical scalars and
17 bags.

## 9. Complete raw UPS field inventory

Legend:

- T: absolute event timestamp in milliseconds;
- C: pre-encoded categorical int64 attribute.

The attribute count includes T. Each sequence also has its agg-only membership
column listed in Section 5.4.

### 9.1 impr: 12 attributes

~~~text
T  impr_x_time
C  impr_x_cat1_id_hn
C  impr_x_cat2_id_hn
C  impr_x_cat3_id_hn
C  impr_x_cat4_id_hn
C  impr_x_cat_id_hn
C  impr_x_goods_id_hn
C  impr_x_mall_id_hn
C  impr_x_page_sn_hn
C  impr_x_sales_hn
C  impr_x_price_hn
C  impr_x_timegap_hn
~~~

### 9.2 clk_long: 12 attributes

~~~text
T  clk_long_x_time
C  clk_long_x_cat1_id_hn
C  clk_long_x_cat2_id_hn
C  clk_long_x_cat3_id_hn
C  clk_long_x_cat4_id_hn
C  clk_long_x_cat_id_hn
C  clk_long_x_goods_id_hn
C  clk_long_x_mall_id_hn
C  clk_long_x_page_sn_hn
C  clk_long_x_sales_hn
C  clk_long_x_price_hn
C  clk_long_x_timegap_hn
~~~

### 9.3 view_long: 30 attributes

~~~text
T  view_long_x_time
C  view_long_x_cat1_id_hn
C  view_long_x_cat2_id_hn
C  view_long_x_cat3_id_hn
C  view_long_x_cat4_id_hn
C  view_long_x_cat_id_hn
C  view_long_x_goods_id_hn
C  view_long_x_mall_id_hn
C  view_long_x_page_sn_hn
C  view_long_x_sales_hn
C  view_long_x_price_hn
C  view_long_x_timegap_hn
C  view_long_x_clk_bottom_img_hn
C  view_long_x_clk_cancel_wish_hn
C  view_long_x_clk_carousel_hn
C  view_long_x_clk_evaluate_hn
C  view_long_x_clk_more_hn
C  view_long_x_clk_svid_hn
C  view_long_x_clk_wish_hn
C  view_long_x_fvid_cv_hn
C  view_long_x_fvid_ratio_hn
C  view_long_x_vid_hn
C  view_long_x_share_hn
C  view_long_x_slide_bottom_detail_hn
C  view_long_x_slide_bottom_img_hn
C  view_long_x_slide_carousel_hn
C  view_long_x_slide_carousel_cnt_hn
C  view_long_x_stay_time_hn
C  view_long_x_switch_sku_hn
C  view_long_x_switch_sku_cnt_hn
~~~

### 9.4 cart_long: 12 attributes

~~~text
T  cart_long_x_time
C  cart_long_x_cat1_id_hn
C  cart_long_x_cat2_id_hn
C  cart_long_x_cat3_id_hn
C  cart_long_x_cat4_id_hn
C  cart_long_x_cat_id_hn
C  cart_long_x_goods_id_hn
C  cart_long_x_mall_id_hn
C  cart_long_x_price_hn
C  cart_long_x_timegap_hn
C  cart_long_x_spec_hn
C  cart_long_x_sku_ids_hn
~~~

### 9.5 buy_long: 13 attributes

~~~text
T  buy_long_x_time
C  buy_long_x_cat1_id_hn
C  buy_long_x_cat2_id_hn
C  buy_long_x_cat3_id_hn
C  buy_long_x_cat4_id_hn
C  buy_long_x_cat_id_hn
C  buy_long_x_goods_id_hn
C  buy_long_x_mall_id_hn
C  buy_long_x_sales_hn
C  buy_long_x_price_hn
C  buy_long_x_timegap_hn
C  buy_long_x_spec_hn
C  buy_long_x_sku_ids_hn
~~~

### 9.6 semi_clk: 6 attributes

~~~text
T  semi_clk_x_time
C  semi_clk_x_cat_id_hn
C  semi_clk_x_goods_id_hn
C  semi_clk_x_mall_id_hn
C  semi_clk_x_page_sn_hn
C  semi_clk_x_timegap_hn
~~~

### 9.7 srch_q2i: 9 attributes

~~~text
T  srch_q2i_x_time
C  srch_q2i_x_cat1_id_hn
C  srch_q2i_x_cat2_id_hn
C  srch_q2i_x_cat3_id_hn
C  srch_q2i_x_cat4_id_hn
C  srch_q2i_x_cat_id_hn
C  srch_q2i_x_goods_id_hn
C  srch_q2i_x_mall_id_hn
C  srch_q2i_x_timegap_hn
~~~

### 9.8 ups_clk_sku: 10 attributes

~~~text
T  ups_clk_sku_x_time
C  ups_clk_sku_x_cat1_id_hn
C  ups_clk_sku_x_cat2_id_hn
C  ups_clk_sku_x_cat3_id_hn
C  ups_clk_sku_x_cat4_id_hn
C  ups_clk_sku_x_cat_id_hn
C  ups_clk_sku_x_goods_id_hn
C  ups_clk_sku_x_mall_id_hn
C  ups_clk_sku_x_timegap_hn
C  ups_clk_sku_x_spec_hn
~~~

### 9.9 flatten_query_hash: 3 attributes

~~~text
T  flatten_query_hash_x_time
C  flatten_query_hash_x_flat_q_hash_hn
C  flatten_query_hash_x_timegap_hn
~~~

## 10. Adapter output contract

After agg expansion or req normalization, one Arrow row represents one
candidate.

| Output family | Current physical Arrow cell type |
|---|---|
| Logical scalar categorical fields | int64 |
| Request-level bag categorical fields | dictionary<int32, list<int64>> |
| Candidate-level bag categorical fields | list<int64> |
| UPS categorical fields | dictionary<int32, list<int64>> |
| Derived time-delta fields | dictionary<int32, list<float32>> |
| scene_id and impr_time | int64 |
| Labels | int64 before tensor conversion |
| candidate_position | int64 |
| search_id / example_ids | Preserved scalar producer type |

The dictionary representation above is enabled by the current
compact_request_lists=true setting. It stores one request-shared list once and
lets all candidates reference it with int32 dictionary indices. Disabling that
setting yields ordinary list<int64> or list<float32> cells with the same
logical values.

The nine adapter-generated continuous sequence columns are:

~~~text
impr_x_time_delta_log1p_seconds
clk_long_x_time_delta_log1p_seconds
view_long_x_time_delta_log1p_seconds
cart_long_x_time_delta_log1p_seconds
buy_long_x_time_delta_log1p_seconds
semi_clk_x_time_delta_log1p_seconds
srch_q2i_x_time_delta_log1p_seconds
ups_clk_sku_x_time_delta_log1p_seconds
flatten_query_hash_x_time_delta_log1p_seconds
~~~

The old seq_len_impr through seq_len_flatten_query_hash fields are not
physical inputs and are not current model categorical features. Sequence
lengths are derived from normalized Arrow offsets.

## 11. Current trusted-input validation policy

The upstream data-team feed is treated as trusted production input.

- HDFS discovery reads Parquet footers and the current configuration compares
  a deterministic sample of file schemas. schema_validation_samples=64 refers
  to file schemas, not 64 payload rows.
- trusted_input=true makes each scanner/rank run detailed raw-contract checks
  on only the first physical row of its first non-empty batch.
- The first non-empty flat output is also checked using only one candidate
  row.
- Later batches skip repeated row-by-row shape, sequence-order, timestamp, and
  label diagnostics.
- validate_prehashed_nonzero=false avoids scanning every encoded field for
  zero on every tensorization batch.
- Complete labels use no mask columns or mask tensor.

This keeps a one-row integration smoke check while removing per-sample
validation from the training hot path.

## 12. Authority and historical-material boundaries

Use the following precedence when documents disagree:

1. Current configs/mdl_rankmixer.yaml and src/dataloader.py behavior.
2. Confirmed CVR decisions consolidated in the relocated
   DATA_ADAPTATION_REVIEW_20260717.md.
3. Historical sample observations in the relocated data format.md.
4. The relocated data.md only for generic agg/req physical-layout concepts.

Specifically, the following older material is informative but not normative:

- CTR labels and the smaller UPS set in data.md;
- old embedding dimensions and 11-times-dimension output calculations;
- historical UPS MAX values as proposed truncation limits;
- seq_len_* as separate categorical inputs;
- assumptions that all nested lists can be flattened identically.

For a future authoritative dump of all 630 upstream fields, export the
Parquet schema from HDFS and append the unprojected column names separately.
Do not add those columns to adapter input_columns unless the model actually
consumes them.

## 13. Per-field physical Arrow format matrix

This section restores the field-level physical types recorded in the relocated
data format.md. “Observed agg/train” and “Observed req/test” describe the two
historical producer snapshots, not tensor shapes and not a promise that every
future partition will retain the same nesting. The current adapter output type
is taken from src/dataloader.py.

The latest HDFS schema summary takes precedence over those historical
snapshots: current _hn columns use only list<int64> and
list<list<int64>>; currency_hn and price_hn were specifically observed as
list<int64>, while nested UPS token fields use list<list<int64>>. Therefore a
historical agg/train entry of list<list<int64>> for a scalar such as
currency_hn is retained below as provenance, not asserted as the current HDFS
type.

For the current HDFS semantic layout, every row marked scalar uses one value
per logical request or candidate and normally has raw type list<int64> at the
aggregate-row boundary. Every row marked bag needs the extra value axis and
normally has raw type list<list<int64>>. The adapter also accepts the
producer-retained singleton axis shown in the historical columns.

The Adapter logical value type column below describes the value after
normalization. With compact_request_lists=true, request bags are physically
dictionary<int32, list<int64>> in the flat Arrow table; candidate bags remain
plain list<int64>.

For request fields, the outer agg list is the request axis. For candidate fields,
the outer list is the candidate axis. An inner list is either a true bag or a
producer-retained singleton feature axis; the Logical kind column decides which.
Consequently, `list<list<int64>>` does not by itself mean “bag”.

All 169 non-sequence fields were recorded as `list<list<int64>>` in the agg/train
snapshot. The req/test snapshot recorded 50 as `list<int64>` and 119 as
`list<list<int64>>`. The exact mapping follows.

### 13.1 Context: 24 request fields

| Field | Logical kind | Current HDFS normal form | Historical agg/train raw type | Historical req/test raw type | Adapter logical value type | Historical note |
|---|---|---|---|---|---|---|
| `currency_hn` | request scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `hash_language_site_hn` | request scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `language_hn` | request scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `page_elsn_hn` | request scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `page_sn_hn` | request scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `plat_hn` | request scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `region_hn` | request scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `scene_id_hn` | request scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `site_id_hn` | request scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `timezone_hn` | request scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `origin_query_hash_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `query_arr_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `query_hash_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `query_terms_hash_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `query_tfidf_term_hash_list_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `query_extend_translation_hash_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `search_method_hn` | request scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `sess_q2q_hash_list_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `recall_merge_cate_levels_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `recall_merge_cate1_ids_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `recall_merge_cate_ids_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `scene_clk_cnt_15d_hit_hn` | request scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `scene_impr_cnt_15d_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `scene_impr_cnt_15d_hit_hn` | request scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |

### 13.2 User: 23 request fields

| Field | Logical kind | Current HDFS normal form | Historical agg/train raw type | Historical req/test raw type | Adapter logical value type | Historical note |
|---|---|---|---|---|---|---|
| `uid_or_bg_hn` | request scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `u_fst_ordr_cnt_mix_d_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `clk_cnt_1d_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `clk_3d_cnt_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `clk_1d_cat_cnt_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `cart_cnt_1d_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `cart_cnt_3d_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `clk_7d_page_sns_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `clk_7d_page_elsns_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `cart_7d_cat1_ids_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `flip_mall_ids_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `list_clk_cat1_ids_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `list_clk_cat_ids_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `ups_in_cart_2h_sku_cur_prices_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `ups_in_cart_goods_hn_share` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `ups_incart_cat1_id_nc_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `ups_in_cart_tg_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `ups_query_term_hash_v2_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `ups_query_tg_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `ups_search_method_hash_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | — |
| `view_30m_cat1_ids_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | Historical req/test non-null: 25,458 / 29,906. |
| `view_7d_page_sns_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | Historical req/test non-null: 28,990 / 29,906. |
| `view_7d_page_elsns_hn` | request bag | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | `list<int64>` | Historical req/test non-null: 28,990 / 29,906. |

### 13.3 Item: 104 candidate fields

| Field | Logical kind | Current HDFS normal form | Historical agg/train raw type | Historical req/test raw type | Adapter logical value type | Historical note |
|---|---|---|---|---|---|---|
| `cat_id_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `cat1_id_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `cat2_id_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `cat3_id_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `cat4_id_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `goods_id_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `goods_name_bigram_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `goods_ner_infos_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `goods_scene_clk_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `goods_title_tfidf_term_hash_list_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `goods_avlb_sku_num_dis_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `goods_onsale_sku_num_dis_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `goods_cluster_id_1w_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `rev_ratings_cnt_crs_pos_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `g_sku_spec_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `g_sku_spec_hash_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `g_sku_spec_unit_list_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `g_prpty_val_id_list_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `sku_id_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `sku_price_v2_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `sku_sales_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `sku_spec_hash_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `sku_spec_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `sku_spec_vids_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `sku_cart_cnt_7d_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `sku_ordr_cnt_1m_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `sku_price_dis_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `sku_sales_dis_hn` | candidate bag | `list<list<int64>>` | `list<list<int64>>` | `list<list<int64>>` | `list<int64>` | — |
| `price_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `price_bef_coupon_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `price_after_promotion_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `price_after_promotion_div_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `mkt_prc_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `show_price_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `is_promotion_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `promotion_discount_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `auto_price_p05_dis` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `auto_price_p10_dis_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `ori_price_hn_share` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | Historical req/test non-null: 29,906 / 29,906. |
| `sales_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `auto_sales_p10_dis` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `c_adj_cart_cvr_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `c_adj_ctr_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `c_adj_ordr_cvr_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `c_cart_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `c_clk_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `c_impr_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `c_ordr_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `c_simi_adj_cart_cvr_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `c_simi_adj_ctr_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `c_simi_cart_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `c_simi_clk_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `c_simi_impr_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `idx_c_adj_cart_cvr_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `idx_c_adj_ctr_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `idx_c_adj_ordr_cvr_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `idx_c_cart_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `idx_c_clk_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `idx_c_impr_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `idx_c_ordr_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `idx_c_simi_adj_cart_cvr_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `idx_c_simi_adj_ctr_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `idx_c_simi_cart_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `idx_c_simi_clk_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `idx_c_simi_impr_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `scene_adj_cartcvr_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `scene_adj_ctr_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `scene_adj_cvr_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `scene_cart_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `nfk_sales_14d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `nfk_price_14d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `nfk_gmv_14d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `i2i2cat2_swing_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `i2i_coclk_hn_share` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `i2i_list_amazoni2ifullgmv_hn_share` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `i2i_list_multimodal_hn_share` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `i2i_list_swingv3gmv_hn_share` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `i2i_hit_site_q2i_idx_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | Historical sample contained several null values. |
| `only_semi_swingi2i_cut60_hn_share` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `semi_swingi2i_cut30_hn_share` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `offline_outside_goods_id_list_hn_share` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | Both agg/train and req/test examples were inspected. |
| `site_q2i_good_list_hn_share` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | The historical table records req/test as list<int64>; verify against a refreshed live schema if the producer changes. |
| `multimodal_i2i_hit_cart_size_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `multimodal_i2i_hit_clk_size_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `main_goods_ids_hn_share` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | Historical req/test non-null: 9,824 / 29,906. |
| `adj_cartcvr_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `adj_ctr_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `adj_cvr_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `buy_long_spec_vids_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `cart_long_spec_vids_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `create_time_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `mall_id_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `sellr_type_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `opt_id_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `site_x_asian_code_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `f_goods_view_times_tg_l1_hn` | candidate scalar; optional scan column | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | Historical sample contained null values; the old document spelled the field f_goods_view_times_tg_11_hn. |
| `target_gs_last_cart_tg_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | Historical examples were all null. |
| `impr_3h_tg_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `impr_all_tg_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | — |
| `impr_clk_6h_cnt_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `clk_long_goods_abs_timegap_1d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | Historical sample contained null values. |
| `impr_long_goods_abs_timegap_1d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `mid_goods_prc_list_dis` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `mid_cmprc_diff_list_dis` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |

### 13.4 Cross: 15 candidate fields

| Field | Logical kind | Current HDFS normal form | Historical agg/train raw type | Historical req/test raw type | Adapter logical value type | Historical note |
|---|---|---|---|---|---|---|
| `rel_score_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `rel_level_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `q_hit_good_correct_unigram_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | Historical sample contained null values. |
| `q2c_cart_15d_hit_val_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `tit_in_top_query_cnt_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `goods_query_emb32v3_cos_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `query_cat_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | Historical req/test non-null: 7,032 / 29,906. |
| `query_pay_cnt_15d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<int64>` | `int64` | Historical req/test non-null: 6,342 / 29,906. |
| `clk_hit_i2i_idx_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `cart_hit_i2i_idx_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `cart_long_hit_samestyle_i2i_idx_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | Historical examples were all null. |
| `ups_clkv2_i2i_goods_ids_hit_size` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `ups_clkv2_i2i_goods_ids_hit_all_size` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `us_ctr_price_dis50_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `impr_cat_clk_goods_ids_cnt_1d_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |

### 13.5 Creative: 3 candidate fields

| Field | Logical kind | Current HDFS normal form | Historical agg/train raw type | Historical req/test raw type | Adapter logical value type | Historical note |
|---|---|---|---|---|---|---|
| `ad_id_bin_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `campaign_id_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |
| `idx_goods_creative_id_hn` | candidate scalar | `list<int64>` | `list<list<int64>>` | `list<list<int64>>` | `int64` | — |

### 13.6 Raw UPS attributes: 107 fields

The historical test.parquet probe recorded every raw UPS attribute as
`list<list<int64>>`: the outer list is the event axis and every inner token list
had length exactly one. The adapter also accepts the equivalent direct
`list<int64>` representation. The table makes that inherited format explicit for
each field.

#### 13.6.1 impr

| Raw field | Value role | Observed raw type | Also accepted | Adapter representation |
|---|---|---|---|---|
| `impr_x_time` | absolute event time in milliseconds | `list<list<int64>>` | `list<int64>` | `impr_x_time_delta_log1p_seconds: list<float32>` (derived; raw time is consumed) |
| `impr_x_cat1_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `impr_x_cat1_id_hn: list<int64>` |
| `impr_x_cat2_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `impr_x_cat2_id_hn: list<int64>` |
| `impr_x_cat3_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `impr_x_cat3_id_hn: list<int64>` |
| `impr_x_cat4_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `impr_x_cat4_id_hn: list<int64>` |
| `impr_x_cat_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `impr_x_cat_id_hn: list<int64>` |
| `impr_x_goods_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `impr_x_goods_id_hn: list<int64>` |
| `impr_x_mall_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `impr_x_mall_id_hn: list<int64>` |
| `impr_x_page_sn_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `impr_x_page_sn_hn: list<int64>` |
| `impr_x_sales_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `impr_x_sales_hn: list<int64>` |
| `impr_x_price_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `impr_x_price_hn: list<int64>` |
| `impr_x_timegap_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `impr_x_timegap_hn: list<int64>` |

#### 13.6.2 clk_long

| Raw field | Value role | Observed raw type | Also accepted | Adapter representation |
|---|---|---|---|---|
| `clk_long_x_time` | absolute event time in milliseconds | `list<list<int64>>` | `list<int64>` | `clk_long_x_time_delta_log1p_seconds: list<float32>` (derived; raw time is consumed) |
| `clk_long_x_cat1_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `clk_long_x_cat1_id_hn: list<int64>` |
| `clk_long_x_cat2_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `clk_long_x_cat2_id_hn: list<int64>` |
| `clk_long_x_cat3_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `clk_long_x_cat3_id_hn: list<int64>` |
| `clk_long_x_cat4_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `clk_long_x_cat4_id_hn: list<int64>` |
| `clk_long_x_cat_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `clk_long_x_cat_id_hn: list<int64>` |
| `clk_long_x_goods_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `clk_long_x_goods_id_hn: list<int64>` |
| `clk_long_x_mall_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `clk_long_x_mall_id_hn: list<int64>` |
| `clk_long_x_page_sn_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `clk_long_x_page_sn_hn: list<int64>` |
| `clk_long_x_sales_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `clk_long_x_sales_hn: list<int64>` |
| `clk_long_x_price_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `clk_long_x_price_hn: list<int64>` |
| `clk_long_x_timegap_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `clk_long_x_timegap_hn: list<int64>` |

#### 13.6.3 view_long

| Raw field | Value role | Observed raw type | Also accepted | Adapter representation |
|---|---|---|---|---|
| `view_long_x_time` | absolute event time in milliseconds | `list<list<int64>>` | `list<int64>` | `view_long_x_time_delta_log1p_seconds: list<float32>` (derived; raw time is consumed) |
| `view_long_x_cat1_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_cat1_id_hn: list<int64>` |
| `view_long_x_cat2_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_cat2_id_hn: list<int64>` |
| `view_long_x_cat3_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_cat3_id_hn: list<int64>` |
| `view_long_x_cat4_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_cat4_id_hn: list<int64>` |
| `view_long_x_cat_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_cat_id_hn: list<int64>` |
| `view_long_x_goods_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_goods_id_hn: list<int64>` |
| `view_long_x_mall_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_mall_id_hn: list<int64>` |
| `view_long_x_page_sn_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_page_sn_hn: list<int64>` |
| `view_long_x_sales_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_sales_hn: list<int64>` |
| `view_long_x_price_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_price_hn: list<int64>` |
| `view_long_x_timegap_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_timegap_hn: list<int64>` |
| `view_long_x_clk_bottom_img_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_clk_bottom_img_hn: list<int64>` |
| `view_long_x_clk_cancel_wish_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_clk_cancel_wish_hn: list<int64>` |
| `view_long_x_clk_carousel_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_clk_carousel_hn: list<int64>` |
| `view_long_x_clk_evaluate_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_clk_evaluate_hn: list<int64>` |
| `view_long_x_clk_more_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_clk_more_hn: list<int64>` |
| `view_long_x_clk_svid_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_clk_svid_hn: list<int64>` |
| `view_long_x_clk_wish_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_clk_wish_hn: list<int64>` |
| `view_long_x_fvid_cv_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_fvid_cv_hn: list<int64>` |
| `view_long_x_fvid_ratio_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_fvid_ratio_hn: list<int64>` |
| `view_long_x_vid_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_vid_hn: list<int64>` |
| `view_long_x_share_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_share_hn: list<int64>` |
| `view_long_x_slide_bottom_detail_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_slide_bottom_detail_hn: list<int64>` |
| `view_long_x_slide_bottom_img_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_slide_bottom_img_hn: list<int64>` |
| `view_long_x_slide_carousel_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_slide_carousel_hn: list<int64>` |
| `view_long_x_slide_carousel_cnt_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_slide_carousel_cnt_hn: list<int64>` |
| `view_long_x_stay_time_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_stay_time_hn: list<int64>` |
| `view_long_x_switch_sku_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_switch_sku_hn: list<int64>` |
| `view_long_x_switch_sku_cnt_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `view_long_x_switch_sku_cnt_hn: list<int64>` |

#### 13.6.4 cart_long

| Raw field | Value role | Observed raw type | Also accepted | Adapter representation |
|---|---|---|---|---|
| `cart_long_x_time` | absolute event time in milliseconds | `list<list<int64>>` | `list<int64>` | `cart_long_x_time_delta_log1p_seconds: list<float32>` (derived; raw time is consumed) |
| `cart_long_x_cat1_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `cart_long_x_cat1_id_hn: list<int64>` |
| `cart_long_x_cat2_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `cart_long_x_cat2_id_hn: list<int64>` |
| `cart_long_x_cat3_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `cart_long_x_cat3_id_hn: list<int64>` |
| `cart_long_x_cat4_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `cart_long_x_cat4_id_hn: list<int64>` |
| `cart_long_x_cat_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `cart_long_x_cat_id_hn: list<int64>` |
| `cart_long_x_goods_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `cart_long_x_goods_id_hn: list<int64>` |
| `cart_long_x_mall_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `cart_long_x_mall_id_hn: list<int64>` |
| `cart_long_x_price_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `cart_long_x_price_hn: list<int64>` |
| `cart_long_x_timegap_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `cart_long_x_timegap_hn: list<int64>` |
| `cart_long_x_spec_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `cart_long_x_spec_hn: list<int64>` |
| `cart_long_x_sku_ids_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `cart_long_x_sku_ids_hn: list<int64>` |

#### 13.6.5 buy_long

| Raw field | Value role | Observed raw type | Also accepted | Adapter representation |
|---|---|---|---|---|
| `buy_long_x_time` | absolute event time in milliseconds | `list<list<int64>>` | `list<int64>` | `buy_long_x_time_delta_log1p_seconds: list<float32>` (derived; raw time is consumed) |
| `buy_long_x_cat1_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `buy_long_x_cat1_id_hn: list<int64>` |
| `buy_long_x_cat2_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `buy_long_x_cat2_id_hn: list<int64>` |
| `buy_long_x_cat3_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `buy_long_x_cat3_id_hn: list<int64>` |
| `buy_long_x_cat4_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `buy_long_x_cat4_id_hn: list<int64>` |
| `buy_long_x_cat_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `buy_long_x_cat_id_hn: list<int64>` |
| `buy_long_x_goods_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `buy_long_x_goods_id_hn: list<int64>` |
| `buy_long_x_mall_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `buy_long_x_mall_id_hn: list<int64>` |
| `buy_long_x_sales_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `buy_long_x_sales_hn: list<int64>` |
| `buy_long_x_price_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `buy_long_x_price_hn: list<int64>` |
| `buy_long_x_timegap_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `buy_long_x_timegap_hn: list<int64>` |
| `buy_long_x_spec_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `buy_long_x_spec_hn: list<int64>` |
| `buy_long_x_sku_ids_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `buy_long_x_sku_ids_hn: list<int64>` |

#### 13.6.6 semi_clk

| Raw field | Value role | Observed raw type | Also accepted | Adapter representation |
|---|---|---|---|---|
| `semi_clk_x_time` | absolute event time in milliseconds | `list<list<int64>>` | `list<int64>` | `semi_clk_x_time_delta_log1p_seconds: list<float32>` (derived; raw time is consumed) |
| `semi_clk_x_cat_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `semi_clk_x_cat_id_hn: list<int64>` |
| `semi_clk_x_goods_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `semi_clk_x_goods_id_hn: list<int64>` |
| `semi_clk_x_mall_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `semi_clk_x_mall_id_hn: list<int64>` |
| `semi_clk_x_page_sn_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `semi_clk_x_page_sn_hn: list<int64>` |
| `semi_clk_x_timegap_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `semi_clk_x_timegap_hn: list<int64>` |

#### 13.6.7 srch_q2i

| Raw field | Value role | Observed raw type | Also accepted | Adapter representation |
|---|---|---|---|---|
| `srch_q2i_x_time` | absolute event time in milliseconds | `list<list<int64>>` | `list<int64>` | `srch_q2i_x_time_delta_log1p_seconds: list<float32>` (derived; raw time is consumed) |
| `srch_q2i_x_cat1_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `srch_q2i_x_cat1_id_hn: list<int64>` |
| `srch_q2i_x_cat2_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `srch_q2i_x_cat2_id_hn: list<int64>` |
| `srch_q2i_x_cat3_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `srch_q2i_x_cat3_id_hn: list<int64>` |
| `srch_q2i_x_cat4_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `srch_q2i_x_cat4_id_hn: list<int64>` |
| `srch_q2i_x_cat_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `srch_q2i_x_cat_id_hn: list<int64>` |
| `srch_q2i_x_goods_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `srch_q2i_x_goods_id_hn: list<int64>` |
| `srch_q2i_x_mall_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `srch_q2i_x_mall_id_hn: list<int64>` |
| `srch_q2i_x_timegap_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `srch_q2i_x_timegap_hn: list<int64>` |

#### 13.6.8 ups_clk_sku

| Raw field | Value role | Observed raw type | Also accepted | Adapter representation |
|---|---|---|---|---|
| `ups_clk_sku_x_time` | absolute event time in milliseconds | `list<list<int64>>` | `list<int64>` | `ups_clk_sku_x_time_delta_log1p_seconds: list<float32>` (derived; raw time is consumed) |
| `ups_clk_sku_x_cat1_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `ups_clk_sku_x_cat1_id_hn: list<int64>` |
| `ups_clk_sku_x_cat2_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `ups_clk_sku_x_cat2_id_hn: list<int64>` |
| `ups_clk_sku_x_cat3_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `ups_clk_sku_x_cat3_id_hn: list<int64>` |
| `ups_clk_sku_x_cat4_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `ups_clk_sku_x_cat4_id_hn: list<int64>` |
| `ups_clk_sku_x_cat_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `ups_clk_sku_x_cat_id_hn: list<int64>` |
| `ups_clk_sku_x_goods_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `ups_clk_sku_x_goods_id_hn: list<int64>` |
| `ups_clk_sku_x_mall_id_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `ups_clk_sku_x_mall_id_hn: list<int64>` |
| `ups_clk_sku_x_timegap_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `ups_clk_sku_x_timegap_hn: list<int64>` |
| `ups_clk_sku_x_spec_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `ups_clk_sku_x_spec_hn: list<int64>` |

#### 13.6.9 flatten_query_hash

| Raw field | Value role | Observed raw type | Also accepted | Adapter representation |
|---|---|---|---|---|
| `flatten_query_hash_x_time` | absolute event time in milliseconds | `list<list<int64>>` | `list<int64>` | `flatten_query_hash_x_time_delta_log1p_seconds: list<float32>` (derived; raw time is consumed) |
| `flatten_query_hash_x_flat_q_hash_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `flatten_query_hash_x_flat_q_hash_hn: list<int64>` |
| `flatten_query_hash_x_timegap_hn` | pre-encoded categorical event attribute | `list<list<int64>>` | `list<int64>` | `flatten_query_hash_x_timegap_hn: list<int64>` |

### 13.7 Partition, metadata, label, and index formats

| Field | agg raw format | req raw format | Adapter output / handling |
|---|---|---|---|
| `pt` | Hive directory token `YYYY-MM-DD` | Same | Partition path component; not assumed to be a Parquet column |
| `hr` | Hive directory token `HH` | Same | Partition path component; not assumed to be a Parquet column |
| `search_id` | `list<T>` over requests | scalar `T` or singleton `list<T>` | Scalar `T` repeated per candidate; producer scalar type is preserved |
| `scene_id` | `list<int64>` over requests | `int64` or singleton `list<int64>` | `int64` per candidate |
| `impr_time` | `list<int64>` over requests | `int64` or singleton `list<int64>` | `int64` milliseconds per candidate |
| `example_ids` | candidate-aligned `list<T>` when present | candidate-aligned `list<T>` when present | Optional scalar `T` per candidate |
| `label_fst_cart` | candidate-aligned `list<int64>` | candidate-aligned `list<int64>` | scalar `int64`, then training `float32` |
| `upid_fst_trgt_noc_clk_pay_24h` | candidate-aligned `list<int64>` | candidate-aligned `list<int64>` | scalar `int64`, then training `float32` |
| `cateid_is_fst_scene_sp_filter` | candidate-aligned `list<int64>` | candidate-aligned `list<int64>` | scalar `int64`, then training `float32` |
| `context_indices` | `list<int64>` | absent | Consumed during agg expansion |
| `target_indices` | `list<int64>` | absent | Consumed during agg expansion |
| `impr_x_indices` | `list<list<int64>>` (event to request IDs) | absent | Consumed during agg UPS membership filtering |
| `clk_long_x_indices` | `list<list<int64>>` (event to request IDs) | absent | Consumed during agg UPS membership filtering |
| `view_long_x_indices` | `list<list<int64>>` (event to request IDs) | absent | Consumed during agg UPS membership filtering |
| `cart_long_x_indices` | `list<list<int64>>` (event to request IDs) | absent | Consumed during agg UPS membership filtering |
| `buy_long_x_indices` | `list<list<int64>>` (event to request IDs) | absent | Consumed during agg UPS membership filtering |
| `semi_clk_x_indices` | `list<list<int64>>` (event to request IDs) | absent | Consumed during agg UPS membership filtering |
| `srch_q2i_x_indices` | `list<list<int64>>` (event to request IDs) | absent | Consumed during agg UPS membership filtering |
| `ups_clk_sku_x_indices` | `list<list<int64>>` (event to request IDs) | absent | Consumed during agg UPS membership filtering |
| `flatten_query_hash_x_indices` | `list<list<int64>>` (event to request IDs) | absent | Consumed during agg UPS membership filtering |
| `candidate_position` | absent | absent | Adapter-generated scalar `int64` |

Here `T` means the producer-defined scalar type. The current adapter deliberately
preserves the scalar type of search_id and example_ids rather than claiming they
are int64 without a live schema.
