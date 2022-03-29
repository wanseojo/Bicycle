#ifdef USE_OPENCV

#include <cfloat>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layers/udb_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/strparam.hpp"
#include "caffe/util/im_transforms.hpp"
#include "caffe/util/path_utils.hpp"
#include <boost/filesystem.hpp>

#ifdef USE_CUDNN
#define MAX_BATCH_FOR_SINGLE_THREAD 0
#else
#define MAX_BATCH_FOR_SINGLE_THREAD 100000
#endif

#define RGB2INT(r, g, b)    ((((int)(r)) << 16) | (((int)(g)) << 8) | ((int)(b)))
#define INT2R(v)            ((unsigned char)(v >> 16))
#define INT2G(v)            ((unsigned char)((v >> 8) & 0xff))
#define INT2B(v)            ((unsigned char)(v & 0xff))

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

	template <typename Dtype>
	BlockingQueue<int> UDBDataLayer<Dtype>::fixed_scale_index_[MAX_UDB_NUM];

	template <typename Dtype>
	int UDBDataLayer<Dtype>::udb_layer_count_ = -1;

	template <typename Dtype>
	const int UDBDataLayer<Dtype>::input_channel_[32] = { 3, 4, 4, 3, 5, 4, 5, 6 };

	template <typename Dtype>
	UDBDataLayer<Dtype>::UDBDataLayer(const LayerParameter& param)
		: Layer<Dtype>(param), prefetch_free_(), prefetch_full_(), prefetch_current_(NULL) {
		UDBDataParameter udb_data_param = this->layer_param_.udb_data_param();

		udb_index_ = udb_layer_count_;
		udb_layer_count_++;
		CHECK(udb_layer_count_ < MAX_UDB_NUM);

		num_classes_ = udb_data_param.num_classes();
		num_objects_ = udb_data_param.num_objects();
		scale_min_ = udb_data_param.scale_min();
		scale_max_ = udb_data_param.scale_max();
		parse_strparam_i(udb_data_param.fixed_scale().c_str(), fixed_scale_);
		multiple_ = udb_data_param.multiple();
		rnd_scale_ = udb_data_param.rnd_scale();
		rnd_crop_ = udb_data_param.rnd_crop();
		unconstrained_rnd_crop_prob_ = udb_data_param.unconstrained_rnd_crop_prob();
		unconstrained_rnd_crop_min_scale_ = udb_data_param.unconstrained_rnd_crop_min_scale();
		rnd_aspect_ = udb_data_param.rnd_aspect();
		rnd_mirror_ = udb_data_param.rnd_mirror();
		rnd_affine_ = udb_data_param.rnd_affine();
		fixed_aspect_ = udb_data_param.fixed_aspect();
		mean0_ = udb_data_param.mean0();
		mean1_ = udb_data_param.mean1();
		mean2_ = udb_data_param.mean2();
		mean_noise_mode_ = udb_data_param.mean_noise_mode();
		mean0_noise_std_ = udb_data_param.mean0_noise_std();
		mean1_noise_std_ = udb_data_param.mean1_noise_std();
		mean2_noise_std_ = udb_data_param.mean2_noise_std();
		mean0_noise_ = Dtype(0);
		mean1_noise_ = Dtype(0);
		mean2_noise_ = Dtype(0);
		use_limit_ = udb_data_param.use_limit();
		max_limit_ = udb_data_param.max_limit();
		min_limit_ = udb_data_param.min_limit();
		parse_strparam_f(udb_data_param.image_roi().c_str(), image_roi_);
		color_aug_ = udb_data_param.color_aug();
		use_fixed_size_ = fixed_scale_.size() > 0;
		num_segments_ = udb_data_param.segment_index_size();
		seg_output_rgb2int_ = udb_data_param.seg_output_rgb2int();
		post_type_ = udb_data_param.post_type();
		use_blank_roi_ = udb_data_param.use_blank_roi();
		DrawDebug::debug_mode = udb_data_param.debug_mode();
		use_maintain_aspect_ratio_ = udb_data_param.use_maintain_aspect_ratio();
		maintain_aspect_ratio_width_ = udb_data_param.maintain_aspect_ratio_width();
		maintain_aspect_ratio_height_ = udb_data_param.maintain_aspect_ratio_height();
		aspect_ratio_ioa_threshold_ = udb_data_param.aspect_ratio_ioa_threshold();
		round_robin_fixed_scale_ = udb_data_param.round_robin_fixed_scale();
		round_robin_fixed_scale_index_ = 0;
		input_type_ = udb_data_param.input_type();
		use_ignore_as_rpn_gt_ = udb_data_param.use_ignore_as_rpn_gt();
		CHECK_EQ(num_segments_, udb_data_param.segment_map_scale_size());
		CHECK_EQ(num_segments_, udb_data_param.segment_label_resize_size());
		CHECK(fixed_scale_.size() == 0 || (!rnd_scale_ && !rnd_aspect_)) << "rnd_scale and rnd_aspect is not allowed with fixed scale";
		CHECK(fixed_scale_.size() > 0 || (fixed_aspect_ == 0 && image_roi_.size() == 0)) << "fixed_aspect and image_roi_ is allowed only with fixed_scale";
		CHECK(!rnd_crop_ || (fixed_aspect_ == 0 && image_roi_.size() == 0)) << "fixed_aspect and image_roi_ is not allowed with rnd_crop";
		CHECK(udb_data_param.batch_size() % udb_data_param.source_list_size() == 0) << "batch_size must be multiple of number of sources";
		CHECK((udb_data_param.batch_size() / udb_data_param.source_list_size()) % (udb_data_param.sequence() + 1) == 0) << "batch_size / # of sources must be multiple of number of sequences";
		CHECK(!udb_data_param.sequence() || udb_data_param.use_blank_roi()) << "sequence must be with use_blank_roi";
		CHECK(!udb_data_param.sequence() || !udb_data_param.rnd_mosaic()) << "sequence is not allowed with rnd_mosaic";
		CHECK(udb_data_param.patch_list_size() == udb_data_param.patch_aug_num_size() || udb_data_param.patch_aug_num_size() == 0);

		// number of grids for ego lanes
		ego_grid_ = udb_data_param.ego_grid();

		// fixed_perspective
		parse_strparam_f(udb_data_param.fixed_perspective().c_str(), fixed_perspective_);
		use_rnd_perspective_ = fixed_perspective_.size() > 0;

		// perspective parameters. Details are in caffe.proto
		use_rnd_pers_scale_ = udb_data_param.rnd_pers_scale();
		pers_scale_min_ = udb_data_param.pers_scale_min();

		use_rnd_pers_center_ = udb_data_param.rnd_pers_center();
		pers_center_max_ = udb_data_param.pers_center_max();



		// flag for closenss
		use_closeness_ = udb_data_param.closeness();

		use_shadow_ = udb_data_param.shadow();
		if (use_shadow_) {
			parse_strparam_f(udb_data_param.shadow_anchor().c_str(), shadow_anchor_);
			CHECK(shadow_anchor_.size() > 0) << "shadow anchor must be defined if you use shadow effects";
		}

		CHECK(use_closeness_ || !use_shadow_) << "shadow must be used with closeness";

		rnd_fp_patch_ = (udb_data_param.false_positive_list_size() > 0) ? true : false;
		if (rnd_fp_patch_) {
			fp_patch_min_n_ = udb_data_param.fp_patch_min_n();
			fp_patch_max_n_ = udb_data_param.fp_patch_max_n();
			fp_patch_min_ratio_ = udb_data_param.fp_patch_min_ratio();
			fp_patch_max_ratio_ = udb_data_param.fp_patch_max_ratio();
			fp_patch_range_x_min_ = udb_data_param.fp_patch_range_x_min();
			fp_patch_range_y_min_ = udb_data_param.fp_patch_range_y_min();
			fp_patch_range_x_max_ = udb_data_param.fp_patch_range_x_max();
			fp_patch_range_y_max_ = udb_data_param.fp_patch_range_y_max();
			CHECK(fp_patch_min_ratio_ <= fp_patch_max_ratio_);
			CHECK(fp_patch_min_n_ <= fp_patch_max_n_);
			CHECK(fp_patch_range_x_min_ <= fp_patch_range_x_max_ && fp_patch_range_y_min_ <= fp_patch_range_y_max_);
		}

		rnd_mosaic_ = udb_data_param.rnd_mosaic();
		if (rnd_mosaic_) {
			mosaic_crop_scale_min_ = udb_data_param.mosaic_crop_scale_min();
			mosaic_crop_scale_max_ = udb_data_param.mosaic_crop_scale_max();
			mosaic_padding_min_ = udb_data_param.mosaic_padding_min();
			mosaic_ioa_threshold_ = udb_data_param.mosaic_ioa_threshold();
			mosaic_prob_ = udb_data_param.mosaic_prob();
			mosaic_boundary_ = int(udb_data_param.mosaic_boundary() / 2);
		}

		for (int i = 0; i < num_segments_; i++) {
			segment_map_scale_.push_back(udb_data_param.segment_map_scale(i));
			segment_label_resize_.push_back(udb_data_param.segment_label_resize(i));
		}

		batchsz_ = udb_data_param.batch_size();
		od_load_batchsz_ = (rnd_mosaic_) ? batchsz_ * 4 : batchsz_;
		shuffle_ = udb_data_param.shuffle();

		vector<vector<int> > segment_color;
		vector<vector<int> > segment_index(num_segments_);

		parse_strparam_i(udb_data_param.segment_color().c_str(), segment_color);
		for (int i = 0; i < num_segments_; i++) {
			parse_strparam_i(udb_data_param.segment_index(i).c_str(), segment_index[i]);
			CHECK_EQ(segment_color.size(), segment_index[i].size());
		}

		segment_map_.resize(num_segments_);
		segment_map_inv_.resize(num_segments_);
		for (int i = 0; i < num_segments_; i++) {
			for (int j = 0; j < segment_color.size(); j++) {
				int color = RGB2INT(segment_color[j][0], segment_color[j][1], segment_color[j][2]);
				segment_map_[i].insert(std::make_pair(color, segment_index[i][j]));
				segment_map_inv_[i].insert(std::make_pair(segment_index[i][j], color));
			}
		}

		for (int i = 0; i < udb_data_param.top_size(); i++) {
			top_types_.push_back(udb_data_param.top(i));
		}

		for (int i = 0; i < batchsz_; i++) {
			vdata_.push_back(new Blob<Dtype>());
		}
		for (int i = 0; i < batchsz_ * num_segments_; i++) {
			vseg_.push_back(new Blob<Dtype>());
		}
		for (int i = 0; i < batchsz_; i++) {
			vedge_.push_back(new Blob<Dtype>());
		}
		for (int i = 0; i < batchsz_; i++) {
			vtlrReg_.push_back(new Blob<Dtype>());
		}
		for (int i = 0; i < od_load_batchsz_; i++) {
			worker_output_.push_back(shared_ptr<db::UDBDatum>(new db::UDBDatum()));
		}
		if (od_load_batchsz_ > MAX_BATCH_FOR_SINGLE_THREAD) {
			for (int i = 0; i < od_load_batchsz_; i++) {
				worker_output_free_.push(worker_output_[i].get());
			}
			run_workers_ = true;
			for (int i = 0; i < od_load_batchsz_; i++) {
				workers_.push_back(boost::thread(boost::bind(&UDBDataLayer<Dtype>::worker_thread, this)));
			}
		}

#ifdef USE_CUDNN
		const int prefetch_size = !DrawDebug::debug_mode ? param.udb_data_param().prefetch() : 1;
#else
		const int prefetch_size = 1;
#endif
		CHECK_GT(prefetch_size, 0);
		prefetch_.resize(prefetch_size);
		for (int i = 0; i < prefetch_.size(); ++i) {
			prefetch_[i].reset(new UDBBatch<Dtype>(num_segments_));
			prefetch_free_.push(prefetch_[i].get());
		}
	}

	template <typename Dtype>
	UDBDataLayer<Dtype>::~UDBDataLayer() {
		run_workers_ = false;
		for (int i = 0; i < workers_.size(); i++) {
			if (workers_[i].joinable()) {
				workers_[i].interrupt();
				try {
					workers_[i].join();
				}
				catch (boost::thread_interrupted&) {
				}
				catch (std::exception& e) {
					LOG(FATAL) << "Thread exception: " << e.what();
				}
			}
			for (int i = 0; i < patch.size(); i++) {
				if (patch[i]->is_open()) {
					patch[i]->close();
					delete patch[i];
				}
			}
		}

		this->StopInternalThread();
		for (int i = 0; i < vdata_.size(); i++) {
			delete vdata_[i];
		}
		for (int i = 0; i < vseg_.size(); i++) {
			delete vseg_[i];
		}
		for (int i = 0; i < vedge_.size(); i++) {
			delete vedge_[i];
		}
		for (int i = 0; i < vtlrReg_.size(); i++) {
			delete vtlrReg_[i];
		}
	}

	template <typename Dtype>
	void UDBDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		top_cnt_ = top.size();
		CHECK_EQ(top_types_.size(), top_cnt_)
			<< "# of top fields in udb_data_param should be equal to actual # of tops";
		int width = 512;
		int height = 512;
		if (this->layer_param_.udb_data_param().has_fixed_scale()) {
			width = (fixed_scale_[0][0] / multiple_) * multiple_;
			height = (fixed_scale_[0][1] / multiple_) * multiple_;
		}
		vector<int> data_shape = { batchsz_, input_channel_[input_type_], height, width };
		vector<int> im_info_shape = { batchsz_, 12 };
		vector<int> gt_rois_shape = { 1, 6 };
		vector<int> gt_rois_quad_shape = { 1, 10 }; // quadrangle gt rois for 4 point detection
		vector<int> scene_lbl_shape = { batchsz_, 3 };
		vector<int> seg_shape = { batchsz_, 1, height, width };
		vector<int> gt_rois_3d_shape = { 1, 10 };
		vector<int> gt_rois_new_3d_shape = { 1, 17 };
		vector<int> gt_rois_ex_shape = { 1, 32 };
		vector<int> edge_shape = { batchsz_, 1, height, width };
		vector<int> dc_rois_shape = { 1, 5 };
		vector<int> hn_rois_shape = { 1, 5 };
		vector<int> hp_rois_shape = { 1, 5 };
		vector<int> tsr_cls_shape = { batchsz_, 1 };
		vector<int> tlr_cls_shape = { batchsz_, 1 };
		vector<int> tlr_blobs_shape = { batchsz_, 1 };
		vector<int> tlr_blobReg_shape = { batchsz_, 6, height, width };
		vector<int> gt_info_shape = { batchsz_, 3, 512 };
		vector<int> failsafe_shape = { batchsz_, 1 };
		vector<int> img_rois_shape = { batchsz_, 5 };
		vector<int> closeness_shape = { batchsz_, 1 };
		vector<int> ego_out_shape = { batchsz_, (int)ego_grid_, 2 };
		vector<int> file_path_shape = { batchsz_, 1024 }; // get udb data file path, which is used to build ego GT files
		vector<int> meta_info_shape = { batchsz_, 6 };
		vector<int> lane_type_label_shape = { batchsz_ };
		vector<int> boundary_type_label_shape = { batchsz_ };

		use_img_ = use_info_ = use_gt_ = use_scene_lbl_ = use_seg_ = use_gt_3d_ = use_gt_new_3d_ = use_gt_ex_ = use_edge_ = use_dc_ = use_hn_ = use_hp_ = use_gt_info_ = use_failsafe_ = use_img_rois_ = use_closeness_ = use_ego_out_ = use_file_path_ = use_tsr_cls_ = use_tlr_cls_ = use_tlr_blobs_ = use_tlr_blobReg_ = use_gt_quad_ = use_meta_info_ = false;
		for (int i = 0, j = 0; i < top_cnt_; i++) {
			switch (top_types_[i]) {
			case UDBDataParameter_TopConfiguration_IMG:
				top[i]->Reshape(data_shape);
				use_img_ = true;
				break;
			case UDBDataParameter_TopConfiguration_IM_INFO:
				top[i]->Reshape(im_info_shape);
				use_info_ = true;
				break;
			case UDBDataParameter_TopConfiguration_GT_ROIS:
				top[i]->Reshape(gt_rois_shape);
				use_gt_ = true;
				break;
			case UDBDataParameter_TopConfiguration_GT_ROIS_QUAD:
				top[i]->Reshape(gt_rois_quad_shape);
				use_gt_quad_ = true;
				break;
			case UDBDataParameter_TopConfiguration_SCENE_LABEL:
				top[i]->Reshape(scene_lbl_shape);
				use_scene_lbl_ = true;
				break;
			case UDBDataParameter_TopConfiguration_META_INFO:
				top[i]->Reshape(meta_info_shape);
				use_meta_info_ = true;
				break;
			case UDBDataParameter_TopConfiguration_SEG:
				if (!segment_label_resize_[j]) {
					top[i]->Reshape(seg_shape);
				}
				else {
					top[i]->Reshape({ seg_shape[0], seg_shape[1], seg_shape[2] / segment_map_scale_[j], seg_shape[3] / segment_map_scale_[j] });
				}
				j++;
				use_seg_ = true;
				break;
			case UDBDataParameter_TopConfiguration_GT_ROIS_3D:
				top[i]->Reshape(gt_rois_3d_shape);
				use_gt_3d_ = true;
				break;
			case UDBDataParameter_TopConfiguration_GT_ROIS_NEW_3D:
				top[i]->Reshape(gt_rois_new_3d_shape);
				use_gt_new_3d_ = true;
				break;
			case UDBDataParameter_TopConfiguration_GT_ROIS_EX:
				top[i]->Reshape(gt_rois_ex_shape);
				use_gt_ex_ = true;
				break;
			case UDBDataParameter_TopConfiguration_EDGE:
				top[i]->Reshape({ edge_shape[0], edge_shape[1], edge_shape[2] / segment_map_scale_[0], edge_shape[3] / segment_map_scale_[0] });
				use_edge_ = true;
				break;
			case UDBDataParameter_TopConfiguration_DC_ROIS:
				top[i]->Reshape(dc_rois_shape);
				use_dc_ = true;
				break;
			case UDBDataParameter_TopConfiguration_HN_ROIS:
				top[i]->Reshape(hn_rois_shape);
				use_hn_ = true;
				break;
			case UDBDataParameter_TopConfiguration_HP_ROIS:
				top[i]->Reshape(hp_rois_shape);
				use_hp_ = true;
				break;
			case UDBDataParameter_TopConfiguration_GT_INFO:
				top[i]->Reshape(gt_info_shape);
				use_gt_info_ = true;
				break;
			case UDBDataParameter_TopConfiguration_FAILSAFE:
				top[i]->Reshape(failsafe_shape);
				use_failsafe_ = true;
				break;
			case UDBDataParameter_TopConfiguration_IMG_ROIS:
				top[i]->Reshape(img_rois_shape);
				use_img_rois_ = true;
				break;
			case UDBDataParameter_TopConfiguration_CLOSENESS:
				top[i]->Reshape(closeness_shape);
				use_closeness_ = true;
				break;
			case UDBDataParameter_TopConfiguration_EGO_OUT:
				top[i]->Reshape(ego_out_shape);
				use_ego_out_ = true;
				break;
			case UDBDataParameter_TopConfiguration_FILE_PATH:
				top[i]->Reshape(file_path_shape);
				use_file_path_ = true;
				break;
			case UDBDataParameter_TopConfiguration_TSR_CLS:
				top[i]->Reshape(tsr_cls_shape);
				use_tsr_cls_ = true;
				break;
			case UDBDataParameter_TopConfiguration_TLR_CLS:
				top[i]->Reshape(tlr_cls_shape);
				use_tlr_cls_ = true;
				break;
			case UDBDataParameter_TopConfiguration_TLR_BLOBS:
				top[i]->Reshape(tlr_blobs_shape);
				use_tlr_blobs_ = true;
			case UDBDataParameter_TopConfiguration_TLR_BLOBREG:
				top[i]->Reshape(tlr_blobReg_shape);
				use_tlr_blobReg_ = true;
			case UDBDataParameter_TopConfiguration_LANE_TYPE_LABEL:
				top[i]->Reshape(lane_type_label_shape);
				use_lane_type_label_ = true;
				break;
			case UDBDataParameter_TopConfiguration_BOUNDARY_TYPE_LABEL:
				top[i]->Reshape(boundary_type_label_shape);
				use_boundary_type_label_ = true;
				break;
			default:
				break;
			}
		}

		CHECK(use_img_) << "You should provide images.";
		CHECK(use_gt_ || (!use_dc_ && !use_hn_ && !use_hp_)) << "DC_ROIS, HN_ROIS and HP_ROIS is only allowed with GT_ROIS.";
		CHECK(use_gt_ || !use_gt_quad_) << "GT_ROIS_QUAD is only allowed with GT_ROIS.";
	}

	template <typename Dtype>
	void UDBDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		UDBDataParameter udb_data_param = this->layer_param_.udb_data_param();
		// initialize patch
		if (patch.size() == 0 && udb_data_param.patch_list_size() > 0) {
			CHECK(udb_data_param.patch_list_size() == udb_data_param.patch_class_index_size());
			patch.resize(udb_data_param.patch_list_size(), NULL);
			patch_pointer.resize(udb_data_param.patch_list_size());
			patch_class_index.resize(udb_data_param.patch_list_size());
			patch_aug_num.resize(udb_data_param.patch_list_size());
			for (int l = 0; l < patch.size(); l++) {
				patch[l] = new std::ifstream();
				patch[l]->open(udb_data_param.patch_list(l).c_str(), std::ios::in | std::ios::binary);
				if (patch[l]->is_open()) {
					int patch_count = 0;
					patch[l]->read((char*)&patch_count, sizeof(int));
					int64_t patch_header_size = sizeof(int) + patch_count * sizeof(int64_t);
					for (int i = 0; i < patch_count; i++) {
						int64_t pointer = 0;
						patch[l]->read((char*)&pointer, sizeof(int64_t));
						patch_pointer[l].push_back(pointer + patch_header_size);
					}
				}
				patch_class_index[l] = udb_data_param.patch_class_index(l);
				patch_aug_num[l] = udb_data_param.patch_list_size() == udb_data_param.patch_aug_num_size() ? udb_data_param.patch_aug_num(l) : 30;
			}
		}

		// initialize UDB
		if (db_.size() == 0) {
			CHECK_GT(udb_data_param.source_list_size(), 0);
			db_.resize(udb_data_param.source_list_size());
			cursor_.resize(udb_data_param.source_list_size());
			for (int i = 0; i < udb_data_param.source_list_size(); i++) {
				char cache_file[512] = { 0, };
				if (Caffe::log_path()[0] && Caffe::model_file()[0])
					sprintf(cache_file, "%s/%s.%s.%s.cache", Caffe::log_path(), Caffe::model_file(), this->layer_param_.name().c_str(), Path::filename(udb_data_param.source_list(i)).c_str());
				db_[i].reset(new db::UDB(udb_data_param, i, cache_file, Caffe::root_solver()));
				db_[i]->Open();
				cursor_[i].reset(db_[i]->NewCursor());
			}

			if (udb_data_param.false_positive_list().size() > 0) {
				char cache_file[512] = { 0, };
				if (Caffe::log_path()[0] && Caffe::model_file()[0])
					sprintf(cache_file, "%s/%s.%s.%s.cache", Caffe::log_path(), Caffe::model_file(), this->layer_param_.name().c_str(), "false_positive");
				false_positive_db_.reset(new db::UDB(udb_data_param, -100001, cache_file, Caffe::root_solver()));
				false_positive_db_->Open();
				false_positive_cursor_.reset(false_positive_db_->NewCursor());
			}
		}
		// initialize load batch thread
		if (prefetch_.size() > 1) {
			if (!is_started()) {
				DLOG(INFO) << "Initializing prefetch";
				StartInternalThread();
				DLOG(INFO) << "Prefetch initialized.";
			}
			if (prefetch_current_) {
				prefetch_free_.push(prefetch_current_);
			}
			prefetch_current_ = prefetch_full_.pop("Waiting for data");
		}
		else {
			prefetch_current_ = prefetch_[0].get();
			load_batch(prefetch_current_);
		}
		for (int i = 0, seg_index = 0; i < top_cnt_; i++) {
			Blob<Dtype>* relevant_blob = NULL;
			switch (top_types_[i]) {
			case UDBDataParameter_TopConfiguration_IMG:
				relevant_blob = &prefetch_current_->data_;
				break;
			case UDBDataParameter_TopConfiguration_IM_INFO:
				relevant_blob = &prefetch_current_->info_;
				break;
			case UDBDataParameter_TopConfiguration_GT_ROIS:
				relevant_blob = &prefetch_current_->gt_rois_;
				break;
			case UDBDataParameter_TopConfiguration_GT_ROIS_QUAD:
				relevant_blob = &prefetch_current_->gt_rois_quad_;
				break;
			case UDBDataParameter_TopConfiguration_SCENE_LABEL:
				relevant_blob = &prefetch_current_->label_;
				break;
			case UDBDataParameter_TopConfiguration_SEG:
				relevant_blob = prefetch_current_->seg_[seg_index++];
				break;
			case UDBDataParameter_TopConfiguration_GT_ROIS_3D:
				relevant_blob = &prefetch_current_->gt_rois_3d_;
				break;
			case UDBDataParameter_TopConfiguration_GT_ROIS_NEW_3D:
				relevant_blob = &prefetch_current_->gt_rois_new_3d_;
				break;
			case UDBDataParameter_TopConfiguration_GT_ROIS_EX:
				relevant_blob = &prefetch_current_->gt_rois_ex_;
				break;
			case UDBDataParameter_TopConfiguration_EDGE:
				relevant_blob = &prefetch_current_->edge_;
				break;
			case UDBDataParameter_TopConfiguration_DC_ROIS:
				relevant_blob = &prefetch_current_->dc_rois_;
				break;
			case UDBDataParameter_TopConfiguration_HN_ROIS:
				relevant_blob = &prefetch_current_->hn_rois_;
				break;
			case UDBDataParameter_TopConfiguration_HP_ROIS:
				relevant_blob = &prefetch_current_->hp_rois_;
				break;
			case UDBDataParameter_TopConfiguration_GT_INFO:
				relevant_blob = &prefetch_current_->gt_info_;
				break;
			case UDBDataParameter_TopConfiguration_FAILSAFE:
				relevant_blob = &prefetch_current_->failsafe_;
				break;
			case UDBDataParameter_TopConfiguration_IMG_ROIS:
				relevant_blob = &prefetch_current_->img_rois_;
				break;
			case UDBDataParameter_TopConfiguration_CLOSENESS:
				relevant_blob = &prefetch_current_->closeness_;
				break;
			case UDBDataParameter_TopConfiguration_EGO_OUT:
				relevant_blob = &prefetch_current_->ego_out_;
				break;
			case UDBDataParameter_TopConfiguration_FILE_PATH:
				relevant_blob = &prefetch_current_->file_path_;
				break;
			case UDBDataParameter_TopConfiguration_TSR_CLS:
				relevant_blob = &prefetch_current_->tsr_cls_;
				break;
			case UDBDataParameter_TopConfiguration_TLR_CLS:
				relevant_blob = &prefetch_current_->tlr_cls_;
				break;
			case UDBDataParameter_TopConfiguration_TLR_BLOBS:
				relevant_blob = &prefetch_current_->tlr_blobs_;
				break;
			case UDBDataParameter_TopConfiguration_TLR_BLOBREG:
				relevant_blob = &prefetch_current_->tlr_blobReg_;
				break;
			case UDBDataParameter_TopConfiguration_META_INFO:
				relevant_blob = &prefetch_current_->meta_;
				break;
			case UDBDataParameter_TopConfiguration_LANE_TYPE_LABEL:
				relevant_blob = &prefetch_current_->lane_type_label_;
				break;
			case UDBDataParameter_TopConfiguration_BOUNDARY_TYPE_LABEL:
				relevant_blob = &prefetch_current_->boundary_type_label_;
				break;
			default:
				break;
			}
			top[i]->ReshapeLike(*relevant_blob);
			top[i]->set_cpu_data(relevant_blob->mutable_cpu_data());
		}
	}

	template <typename Dtype>
	void UDBDataLayer<Dtype>::InternalThreadEntry() {
		try {
			while (!must_stop()) {
				UDBBatch<Dtype>* batch = prefetch_free_.pop();
				load_batch(batch);
				prefetch_full_.push(batch);
			}
		}
		catch (boost::thread_interrupted&) {
			// Interrupted exception is expected on shutdown
		}
	}

	template <typename Dtype>
	void UDBDataLayer<Dtype>::load_batch(UDBBatch<Dtype>* batch) {
#ifdef _DEBUG
		double read_time = 0;
		double trans_time = 0;
		CPUTimer timer;
		CPUTimer batch_timer;

		batch_timer.Start();
#endif
		// Reshape according to the first datum of each batch
		// on single input batches allows for inputs of varying dimension.

		int load_batchsz_ = od_load_batchsz_;
		int mosaic_n = 0;
		if (rnd_mosaic_) {
			for (int i = 0; i < batchsz_; i++) {
				double random_prob = ((double)rand() / ((double)(RAND_MAX)+1));
				if (random_prob < mosaic_prob_)
					mosaic_n++;
			}
			load_batchsz_ = batchsz_ + mosaic_n * 3;
		}
		vector<db::UDBDatum*> cur_batch_data(load_batchsz_);

		// to get file_path
		if (use_file_path_) {
			batch->file_path_.Reshape(load_batchsz_, 1024);
		}

		bool succ = false;
		while (!succ) {
			// load all data
#ifdef _DEBUG
			timer.Start();
#endif
			if (load_batchsz_ > MAX_BATCH_FOR_SINGLE_THREAD) {
				for (int batch_index = 0; batch_index < load_batchsz_; ++batch_index) {
					const int cursor_index = batch_index / (load_batchsz_ / cursor_.size());
					const db::UDBPoint* point = cursor_[cursor_index]->GetPoint();
					cursor_[cursor_index]->Next();
					worker_input_.push(std::make_pair(point, batch_index));

					// get file_path
					if (use_file_path_) {
						for (int k = 0; k < point->img_path_.size(); k++)
							batch->file_path_.mutable_cpu_data()[batch_index * batch->file_path_.shape(1) + k] = (Dtype)point->img_path_.c_str()[k];
						batch->file_path_.mutable_cpu_data()[batch_index * batch->file_path_.shape(1) + point->img_path_.size()] = 0;
					}
				}
				for (int batch_index = 0; batch_index < load_batchsz_; ++batch_index) {
					db::UDBDatum* dataum = worker_output_full_.pop();
					cur_batch_data[dataum->batch_index_] = dataum;
				}
			}
			else {
				for (int batch_index = 0; batch_index < load_batchsz_; ++batch_index) {
					const int cursor_index = batch_index / (load_batchsz_ / cursor_.size());
					const db::UDBPoint* point = cursor_[cursor_index]->GetPoint();
					cursor_[cursor_index]->Next();
					cur_batch_data[batch_index] = worker_output_[batch_index].get();
					cur_batch_data[batch_index]->batch_index_ = batch_index;
					get_datum(point, cur_batch_data[batch_index]);

					// get file_path
					if (use_file_path_) {
						for (int k = 0; k < point->img_path_.size(); k++)
							batch->file_path_.mutable_cpu_data()[batch_index * batch->file_path_.shape(1) + k] = (Dtype)point->img_path_.c_str()[k];
						batch->file_path_.mutable_cpu_data()[batch_index * batch->file_path_.shape(1) + point->img_path_.size()] = 0;
					}

				}
			}
#ifdef _DEBUG
			read_time += timer.MicroSeconds();

			// transform them all
			timer.Start();
#endif
			succ = TransformData(cur_batch_data, batch);
#ifdef _DEBUG
			trans_time += timer.MicroSeconds();
#endif

			if (load_batchsz_ > MAX_BATCH_FOR_SINGLE_THREAD) {
				for (int batch_index = 0; batch_index < load_batchsz_; ++batch_index) {
					worker_output_free_.push(cur_batch_data[batch_index]);
				}
			}
#ifdef _DEBUG
			timer.Stop();
			batch_timer.Stop();
			DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
			DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
			DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
#endif
			if (!succ) {
				LOG(INFO) << "Retry load_batch()";
			}
		}
	}

	template <typename Dtype>
	void UDBDataLayer<Dtype>::get_datum(const db::UDBPoint* point, db::UDBDatum* udb_datum) {
		std::string img_data, seg_data;

		if (point->img_path_.size()) {
			boost::mutex::scoped_lock lock(tar_reader_mutex_);
			point->tar_reader_img_->read(point->img_path_, img_data);
		}
		if (point->seg_path_.size()) {
			boost::mutex::scoped_lock lock(tar_reader_mutex_);
			point->tar_reader_seg_->read(point->seg_path_, seg
				_data);
		}

		if (img_data.size()) {
			cv::Mat buf(1, img_data.size(), CV_8UC1, const_cast<char*>(img_data.c_str()));
			udb_datum->img_ = cv::imdecode(buf, CV_LOAD_IMAGE_COLOR);
			if (point->ann_path_.size()) {
				if (!(use_lane_type_label_ || use_boundary_type_label_)) {
					CHECK_EQ(udb_datum->img_.rows, point->img_height_) << "Image rows is different between: " << point->img_path_ << "(" << udb_datum->img_.rows << ")<=>" << point->ann_path_ << "(" << point->img_height_ << ")";
					CHECK_EQ(udb_datum->img_.cols, point->img_width_) << "Image cols is different between: " << point->img_path_ << "(" << udb_datum->img_.cols << ")<=>" << point->ann_path_ << "(" << point->img_width_ << ")";
				}
			}
		}
		else {
			udb_datum->img_.release();
		}

		if (seg_data.size()) {
			cv::Mat buf(1, seg_data.size(), CV_8UC1, const_cast<char*>(seg_data.c_str()));
			udb_datum->seg_ = cv::imdecode(buf, CV_LOAD_IMAGE_COLOR);
			CHECK_EQ(udb_datum->img_.rows, udb_datum->seg_.rows) << "Image rows is different between: " << point->img_path_ << "<=>" << point->seg_path_;
			CHECK_EQ(udb_datum->img_.cols, udb_datum->seg_.cols) << "Image cols is different between: " << point->img_path_ << "<=>" << point->seg_path_;
		}
		else {
			udb_datum->seg_.release();
		}

		point->copy(udb_datum);
	}


	template <typename Dtype>
	void UDBDataLayer<Dtype>::worker_thread() {
		while (run_workers_) {
			std::pair<const db::UDBPoint*, int> worker_input = worker_input_.pop();
			const db::UDBPoint* point = worker_input.first;
			db::UDBDatum* udb_datum = worker_output_free_.pop();
			udb_datum->batch_index_ = worker_input.second;

			get_datum(point, udb_datum);

			worker_output_full_.push(udb_datum);
		}
	}

	template<typename Dtype>
	static void Copy32FC3MatToBlob(cv::Mat& img, Dtype* ptr, int ptr_h, int ptr_w) {
		CHECK(img.type() == CV_32FC3);
		int img_w = img.cols;
		int img_h = img.rows;
		int idx = 0;

		Dtype* ptr0 = ptr;
		Dtype* ptr1 = ptr0 + ptr_h * ptr_w;
		Dtype* ptr2 = ptr1 + ptr_h * ptr_w;
		float* srcptr = img.ptr<float>();

		for (int i = 0; i < img_h; i++) {
			for (int j = 0; j < img_w; j++) {
				idx = 3 * (i * img_w + j);
				ptr0[i * ptr_w + j] = srcptr[idx];
				ptr1[i * ptr_w + j] = srcptr[idx + 1];
				ptr2[i * ptr_w + j] = srcptr[idx + 2];
			}
		}
	}

	template<typename Dtype>
	static void Copy32FC1MatToBlob(cv::Mat& img, Dtype* ptr, int ptr_h, int ptr_w) {
		CHECK(img.type() == CV_32FC1);
		int img_w = img.cols;
		int img_h = img.rows;
		int idx = 0;

		Dtype* ptr0 = ptr;
		float* srcptr = img.ptr<float>();

		for (int i = 0; i < img_h; i++) {
			for (int j = 0; j < img_w; j++) {
				idx = (i * img_w + j);
				ptr0[i * ptr_w + j] = srcptr[idx];
			}
		}
	}

	template<typename Dtype>
	static inline void copy_array(Dtype* dest, const float* src, int count) {
		memcpy(dest, src, count * sizeof(Dtype));
	}

	template<typename Dtype>
	static inline void affine_rect(Dtype* rect, const Dtype* affine_param, int width, int height) {
		const Dtype& a = affine_param[0];
		const Dtype& b = affine_param[1];
		const Dtype& c = affine_param[2];
		const Dtype& d = affine_param[3];
		const Dtype& ai = affine_param[4];
		const Dtype& bi = affine_param[5];
		const Dtype& ci = affine_param[6];
		const Dtype& di = affine_param[7];
		Dtype x1 = a * rect[0] + b * rect[1];
		Dtype y1 = c * rect[0] + d * rect[1];
		Dtype x2 = a * rect[2] + b * rect[1];
		Dtype y2 = c * rect[2] + d * rect[1];
		Dtype x3 = a * rect[0] + b * rect[3];
		Dtype y3 = c * rect[0] + d * rect[3];
		Dtype x4 = a * rect[2] + b * rect[3];
		Dtype y4 = c * rect[2] + d * rect[3];
		rect[0] = min<Dtype>(min<Dtype>(x1, x2), min<Dtype>(x3, x4));
		rect[1] = min<Dtype>(min<Dtype>(y1, y2), min<Dtype>(y3, y4));
		rect[2] = max<Dtype>(max<Dtype>(x1, x2), max<Dtype>(x3, x4));
		rect[3] = max<Dtype>(max<Dtype>(y1, y2), max<Dtype>(y3, y4));
		rect[0] = std::max<Dtype>(std::min<Dtype>(rect[0], width - 1), 0);
		rect[1] = std::max<Dtype>(std::min<Dtype>(rect[1], height - 1), 0);
		rect[2] = std::max<Dtype>(std::min<Dtype>(rect[2], width - 1), 0);
		rect[3] = std::max<Dtype>(std::min<Dtype>(rect[3], height - 1), 0);
	}

	template<typename Dtype>
	static inline void affine_point(Dtype* point, const Dtype* affine_param, int width, int height) {
		const Dtype& a = affine_param[0];
		const Dtype& b = affine_param[1];
		const Dtype& c = affine_param[2];
		const Dtype& d = affine_param[3];
		const Dtype& ai = affine_param[4];
		const Dtype& bi = affine_param[5];
		const Dtype& ci = affine_param[6];
		const Dtype& di = affine_param[7];
		Dtype x = a * point[0] + b * point[1];
		Dtype y = c * point[0] + d * point[1];
		point[0] = std::max<Dtype>(std::min<Dtype>(x, width - 1), 0);
		point[1] = std::max<Dtype>(std::min<Dtype>(y, height - 1), 0);
	}

	template<typename Dtype>
	static inline void mirror_rect(Dtype* rect, int width) {
		Dtype oldx1 = rect[0];
		Dtype oldx2 = rect[2];
		rect[0] = width - oldx2 - 1;
		rect[2] = width - oldx1 - 1;
	}

	template<typename Dtype>
	static inline void mirror_points(Dtype* point1, Dtype* point2, int width) {
		Dtype oldx1 = point1[0];
		Dtype oldy1 = point1[1];
		Dtype oldx2 = point2[0];
		Dtype oldy2 = point2[1];
		point1[0] = width - oldx2 - 1;
		point1[1] = oldy2;
		point2[0] = width - oldx1 - 1;
		point2[1] = oldy1;
	}

	template<typename Dtype>
	static inline void mirror_quad(Dtype* quad, int width) {
		mirror_points(&quad[0], &quad[2], width);
		mirror_points(&quad[4], &quad[6], width);
	}


	template<typename Dtype>
	static inline void mirror_new_3d(Dtype* shape3d, int width) {
		const int mirror_direction[10] = { -1, 3, 2, 1, 6, 5, 4, 9, 8, 7 };
		if (shape3d[14] != -1) {
			shape3d[14] = mirror_direction[(int)shape3d[14]];
		}
		if (shape3d[0] != -1) {
			if (shape3d[15] != -1) {
				mirror_points(&shape3d[0], &shape3d[2], width);
				mirror_points(&shape3d[4], &shape3d[6], width);
			}
			switch ((int)shape3d[15]) {
			case 1:
				shape3d[8] = width - shape3d[8] - 1;
				shape3d[10] = width - shape3d[10] - 1;
				shape3d[15] = 2;
				break;
			case 2:
				shape3d[8] = width - shape3d[8] - 1;
				shape3d[10] = width - shape3d[10] - 1;
				shape3d[15] = 1;
				break;
			case 3:
				mirror_points(&shape3d[8], &shape3d[10], width);
			case 4:
				mirror_points(&shape3d[8], &shape3d[12], width);
				shape3d[10] = width - shape3d[10] - 1;
				shape3d[15] = 5;
				break;
			case 5:
				mirror_points(&shape3d[12], &shape3d[8], width);
				shape3d[10] = width - shape3d[10] - 1;
				shape3d[15] = 4;
				break;
			}
		}
	}

	template<typename Dtype>
	static inline void set_min_max(const Dtype* points, int count, int& min_x, int& min_y, int& max_x, int& max_y) {
		for (int i = 0; i < count; i++) {
			if (points[i * 2 + 0] != -1) {
				min_x = std::min<Dtype>(points[i * 2 + 0], min_x);
				max_x = std::max<Dtype>(points[i * 2 + 0], max_x);
				min_y = std::min<Dtype>(points[i * 2 + 1], min_y);
				max_y = std::max<Dtype>(points[i * 2 + 1], max_y);
			}
		}
	}

	template<typename Dtype>
	static inline void transform_points(Dtype* points, int count, int crop_x, int crop_y, float im_scale_x, float im_scale_y) {
		for (int i = 0; i < count; i++) {
			if (points[i * 2 + 0] != -1) {
				points[i * 2 + 0] = (points[i * 2 + 0] - crop_x) * im_scale_x;
				points[i * 2 + 1] = (points[i * 2 + 1] - crop_y) * im_scale_y;
			}
		}
	}

	template<typename Dtype>
	void UDBDataLayer<Dtype>::TransformWithGT(const vector<db::ObjectGT>& vobject_gt, const std::vector<db::Box2D>& vbox_2d, const std::vector<db::Box3D>& vbox_3d, const std::vector<db::BoxNew3D>& vbox_new_3d, const std::vector<db::BoxAttribute>& vbox_attribute, const vector<db::Quad2D>& vquad_2d,
		Dtype* gt_rois, Dtype* gt_rois_3d, Dtype* gt_rois_new_3d, Dtype* gt_rois_ex, Dtype* dc_rois, Dtype* hn_rois, Dtype* hp_rois, Dtype* gt_rois_quad,
		int batch_index, bool mirror, int width, int height,
		int& resized_width, int& resized_height, int& crop_x, int& crop_y, int& crop_w, int& crop_h, Dtype& im_scale_x, Dtype& im_scale_y, Dtype* affine_param,
		int& num_gts, int& num_gts_3d, int& num_gts_new_3d, int& num_gts_ex, int& num_dcs, int& num_hns, int& num_hps, int& num_gts_quad, Dtype* closeness) {

		Dtype& a = affine_param[0];
		Dtype& b = affine_param[1];
		Dtype& c = affine_param[2];
		Dtype& d = affine_param[3];
		Dtype& ai = affine_param[4];
		Dtype& bi = affine_param[5];
		Dtype& ci = affine_param[6];
		Dtype& di = affine_param[7];

		num_gts = num_dcs = num_hns = num_hps = num_gts_3d = num_gts_new_3d = num_gts_ex = num_gts_quad = 0;
		for (int i = 0; i < vobject_gt.size(); i++) {
			const db::Box2D* box_2d = vobject_gt[i].get_box_2d(vbox_2d);
			const db::Quad2D* quad_2d = vobject_gt[i].get_quad_2d(vquad_2d);
			const db::Box3D* box_3d = vobject_gt[i].get_box_3d(vbox_3d);
			const db::BoxNew3D* box_new_3d = vobject_gt[i].get_box_new_3d(vbox_new_3d);
			const db::BoxAttribute* box_attribute = vobject_gt[i].get_box_attribute(vbox_attribute);
			if (box_2d) {
				if (box_2d->label == -100000) {
					if (dc_rois) {
						dc_rois[num_dcs * 5 + 0] = batch_index;
						copy_array(&dc_rois[num_dcs * 5 + 1], box_2d->p, 4);
						num_dcs++;
					}
					if (gt_rois && use_ignore_as_rpn_gt_) {
						gt_rois[num_gts * 6 + 0] = batch_index;
						copy_array(&gt_rois[num_gts * 6 + 1], box_2d->p, 4);
						gt_rois[num_gts * 6 + 5] = box_2d->label;
						num_gts++;
					}
				}
				else if (box_2d->label == -100001) {
					if (hn_rois) {
						hn_rois[num_hns * 5 + 0] = batch_index;
						copy_array(&hn_rois[num_hns * 5 + 1], box_2d->p, 4);
						num_hns++;
					}
				}
				else if (box_2d->label == -100002) {
					if (hp_rois) {
						hp_rois[num_hps * 5 + 0] = batch_index;
						copy_array(&hp_rois[num_hps * 5 + 1], box_2d->p, 4);
						num_hps++;
					}
				}
				else {
					if (gt_rois) {
						gt_rois[num_gts * 6 + 0] = batch_index;
						copy_array(&gt_rois[num_gts * 6 + 1], box_2d->p, 4);
						gt_rois[num_gts * 6 + 5] = box_2d->label;
						num_gts++;
						if (gt_rois_quad && quad_2d) {
							gt_rois_quad[num_gts_quad * 10 + 0] = batch_index;
							copy_array(&gt_rois_quad[num_gts_quad * 10 + 1], quad_2d->p, 8);
							gt_rois_quad[num_gts_quad * 10 + 9] = quad_2d->label;
							num_gts_quad++;
						}
					}
					if (dc_rois && box_2d->label < 0) {
						dc_rois[num_dcs * 5 + 0] = batch_index;
						copy_array(&dc_rois[num_dcs * 5 + 1], box_2d->p, 4);
						num_dcs++;
					}
				}
			}
			if (gt_rois_3d && box_3d) {
				gt_rois_3d[num_gts_3d * 10 + 0] = batch_index;
				copy_array(&gt_rois_3d[num_gts_3d * 10 + 1], box_3d->p, 8);
				gt_rois_3d[num_gts_3d * 10 + 9] = box_3d->direction;
				num_gts_3d++;
			}
			if (gt_rois_new_3d && box_new_3d) {
				gt_rois_new_3d[num_gts_new_3d * 17 + 0] = batch_index;
				copy_array(&gt_rois_new_3d[num_gts_new_3d * 17 + 1], box_new_3d->p, 14);
				gt_rois_new_3d[num_gts_new_3d * 17 + 15] = box_new_3d->direction;
				gt_rois_new_3d[num_gts_new_3d * 17 + 16] = box_new_3d->shape;
				num_gts_new_3d++;
			}
			if (gt_rois_ex && post_type_ == POST_V1 && (box_2d || box_3d)) {
				gt_rois_ex[num_gts_ex * 32 + 0] = batch_index;
				for (int i = 1; i < 32; i++)
					gt_rois_ex[num_gts_ex * 32 + i] = -1;
				if (box_2d) {
					copy_array(&gt_rois_ex[num_gts_ex * 32 + 1], box_2d->p, 4);
					gt_rois_ex[num_gts_ex * 32 + 5] = box_2d->label;
				}
				if (box_attribute) {
					gt_rois_ex[num_gts_ex * 32 + 6] = box_attribute->occluded;
					gt_rois_ex[num_gts_ex * 32 + 7] = box_attribute->truncated;
					if (!box_3d) {
						gt_rois_ex[num_gts_ex * 32 + 24] = box_attribute->direction;
					}
				}
				if (box_3d) {
					copy_array(&gt_rois_ex[num_gts_ex * 32 + 16], box_3d->p, 8);
					gt_rois_ex[num_gts_ex * 32 + 24] = box_3d->direction;
				}
				num_gts_ex++;
			}
			else if (gt_rois_ex && post_type_ == POST_V2 && (box_2d || box_new_3d)) {
				gt_rois_ex[num_gts_ex * 32 + 0] = batch_index;
				for (int i = 1; i < 32; i++)
					gt_rois_ex[num_gts_ex * 32 + i] = -1;
				if (box_2d) {
					copy_array(&gt_rois_ex[num_gts_ex * 32 + 1], box_2d->p, 4);
					gt_rois_ex[num_gts_ex * 32 + 5] = box_2d->label;
				}
				if (box_attribute) {
					gt_rois_ex[num_gts_ex * 32 + 6] = box_attribute->occluded;
					gt_rois_ex[num_gts_ex * 32 + 7] = box_attribute->truncated;
					if (!box_new_3d) {
						gt_rois_ex[num_gts_ex * 32 + 30] = box_attribute->direction;
					}
				}
				if (box_new_3d) {
					copy_array(&gt_rois_ex[num_gts_ex * 32 + 16], box_new_3d->p, 14);
					gt_rois_ex[num_gts_ex * 32 + 30] = box_new_3d->direction;
					gt_rois_ex[num_gts_ex * 32 + 31] = box_new_3d->shape;
				}
				num_gts_ex++;
			}
			else if (gt_rois_ex && post_type_ >= POST_V3 && (box_2d || box_new_3d)) {
				gt_rois_ex[num_gts_ex * 32 + 0] = batch_index;
				for (int i = 1; i < 32; i++)
					gt_rois_ex[num_gts_ex * 32 + i] = -1;
				if (box_2d) {
					copy_array(&gt_rois_ex[num_gts_ex * 32 + 1], box_2d->p, 4);
					gt_rois_ex[num_gts_ex * 32 + 5] = box_2d->label;
				}
				if (box_attribute) {
					gt_rois_ex[num_gts_ex * 32 + 6] = box_attribute->occluded;
					gt_rois_ex[num_gts_ex * 32 + 7] = box_attribute->truncated;
					gt_rois_ex[num_gts_ex * 32 + 8] = box_attribute->age;
					gt_rois_ex[num_gts_ex * 32 + 9] = box_attribute->gender;
					gt_rois_ex[num_gts_ex * 32 + 10] = box_attribute->sit_stand;
					if (!box_new_3d) {
						gt_rois_ex[num_gts_ex * 32 + 30] = box_attribute->direction;
					}
				}
				if (box_new_3d) {
					copy_array(&gt_rois_ex[num_gts_ex * 32 + 16], box_new_3d->p, 14);
					gt_rois_ex[num_gts_ex * 32 + 30] = box_new_3d->direction;
					gt_rois_ex[num_gts_ex * 32 + 31] = box_new_3d->shape;
				}
				num_gts_ex++;
			}
		}

		if (rnd_affine_) {
			Dtype det = 0;
			while (det == 0) {
				a = (1.1 - 0.2 * ((Dtype)caffe_rng_rand() / UINT_MAX));
				b = (0.1 - 0.2 * ((Dtype)caffe_rng_rand() / UINT_MAX));
				c = (0.1 - 0.2 * ((Dtype)caffe_rng_rand() / UINT_MAX));
				d = (1.1 - 0.2 * ((Dtype)caffe_rng_rand() / UINT_MAX));
				det = 1 / (a * d - b * c);
			}
			ai = det * d;
			bi = det * -b;
			ci = det * -c;
			di = det * a;

			if (gt_rois) {
				for (int i = 0; i < num_gts; i++) {
					affine_rect(&gt_rois[i * 6 + 1], affine_param, width, height);
				}
			}

			if (gt_rois_3d) {
				for (int i = 0; i < num_gts_3d; i++) {
					affine_rect(&gt_rois_3d[i * 10 + 1], affine_param, width, height);
					affine_rect(&gt_rois_3d[i * 10 + 5], affine_param, width, height);
				}
			}

			if (gt_rois_new_3d) {
				for (int i = 0; i < num_gts_new_3d; i++) {
					for (int j = 0; j < 7; j++) {
						if (gt_rois_new_3d[i * 17 + 1 + j * 2] != -1)
							affine_point(&gt_rois_new_3d[i * 17 + 1 + j * 2], affine_param, width, height);
					}
				}
			}

			if (gt_rois_ex) {
				for (int i = 0; i < num_gts_ex; i++) {
					if (gt_rois_ex[i * 32 + 1] != -1) {
						affine_rect(&gt_rois_ex[i * 32 + 1], affine_param, width, height);
					}
					if (gt_rois_ex[i * 32 + 16] != -1) {
						if (post_type_ == POST_V1) {
							affine_rect(&gt_rois_ex[i * 32 + 16], affine_param, width, height);
							affine_rect(&gt_rois_ex[i * 32 + 20], affine_param, width, height);
						}
						else if (post_type_ > POST_V1) {
							for (int j = 0; j < 7; j++) {
								if (gt_rois_ex[i * 32 + 16 + j * 2] != -1)
									affine_point(&gt_rois_ex[i * 32 + 16 + j * 2], affine_param, width, height);
							}
						}
					}
				}
			}

			if (dc_rois) {
				for (int i = 0; i < num_dcs; i++) {
					affine_rect(&dc_rois[i * 5 + 1], affine_param, width, height);
				}
			}

			if (hn_rois) {
				for (int i = 0; i < num_hns; i++) {
					affine_rect(&hn_rois[i * 5 + 1], affine_param, width, height);
				}
			}

			if (hp_rois) {
				for (int i = 0; i < num_hps; i++) {
					affine_rect(&hp_rois[i * 5 + 1], affine_param, width, height);
				}
			}
		}

		if (mirror) {
			const int mirror_direction[9] = { -1, 1, 8, 7, 6, 5, 4, 3, 2 };
			if (gt_rois) {
				for (int i = 0; i < num_gts; i++) {
					mirror_rect(&gt_rois[i * 6 + 1], width);
				}
				if (gt_rois_quad) {
					for (int i = 0; i < num_gts_quad; i++) {
						mirror_quad(&gt_rois_quad[i * 10 + 1], width);
					}
					//for (int i = 0; i < num_gts_quad; i++) {
					//  mirror_rect(&gt_rois_quad[i * 10 + 1], width);
					//  mirror_rect(&gt_rois_quad[i * 10 + 5], width);
					//}
				}
			}

			if (gt_rois_3d) {
				for (int i = 0; i < num_gts_3d; i++) {
					mirror_rect(&gt_rois_3d[i * 10 + 1], width);
					mirror_rect(&gt_rois_3d[i * 10 + 5], width);
					gt_rois_3d[i * 10 + 9] = mirror_direction[(int)gt_rois_3d[i * 10 + 9]];
				}
			}

			if (gt_rois_new_3d) {
				for (int i = 0; i < num_gts_new_3d; i++) {
					mirror_new_3d(&gt_rois_new_3d[i * 17 + 1], width);
				}
			}

			if (gt_rois_ex) {
				for (int i = 0; i < num_gts_ex; i++) {
					if (gt_rois_ex[i * 32 + 1] != -1) {
						mirror_rect(&gt_rois_ex[i * 32 + 1], width);
					}
					if (post_type_ == POST_V1) {
						if (gt_rois_ex[i * 32 + 16] != -1) {
							mirror_rect(&gt_rois_ex[i * 32 + 16], width);
							mirror_rect(&gt_rois_ex[i * 32 + 20], width);
						}
						if (gt_rois_ex[i * 32 + 24] != -1) {
							gt_rois_ex[i * 32 + 24] = mirror_direction[(int)gt_rois_ex[i * 32 + 24]];
						}
					}
					else if (post_type_ > POST_V1) {
						mirror_new_3d(&gt_rois_ex[i * 32 + 16], width);
					}
				}
			}

			if (dc_rois) {
				for (int i = 0; i < num_dcs; i++) {
					mirror_rect(&dc_rois[i * 5 + 1], width);
				}
			}

			if (hn_rois) {
				for (int i = 0; i < num_hns; i++) {
					mirror_rect(&hn_rois[i * 5 + 1], width);
				}
			}

			if (hp_rois) {
				for (int i = 0; i < num_hps; i++) {
					mirror_rect(&hp_rois[i * 5 + 1], width);
				}
			}
		}

		crop_x = 0;
		crop_y = 0;
		crop_w = width;
		crop_h = height;

		int min_x = INT_MAX;
		int min_y = INT_MAX;
		int max_x = INT_MIN;
		int max_y = INT_MIN;

		if (gt_rois && num_gts) {
			for (int i = 0; i < num_gts; i++) {
				set_min_max(&gt_rois[i * 6 + 1], 2, min_x, min_y, max_x, max_y);
			}
			if (gt_rois_quad && num_gts_quad) {
				for (int i = 0; i < num_gts_quad; i++) {
					set_min_max(&gt_rois_quad[i * 10 + 1], 4, min_x, min_y, max_x, max_y);
				}
			}
		}

		if (gt_rois_3d && num_gts_3d) {
			for (int i = 0; i < num_gts_3d; i++) {
				set_min_max(&gt_rois_3d[i * 10 + 1], 4, min_x, min_y, max_x, max_y);
			}
		}

		if (gt_rois_new_3d && num_gts_new_3d) {
			for (int i = 0; i < num_gts_new_3d; i++) {
				set_min_max(&gt_rois_new_3d[i * 17 + 1], 7, min_x, min_y, max_x, max_y);
			}
		}

		if (gt_rois_ex && num_gts_ex) {
			for (int i = 0; i < num_gts_ex; i++) {
				if (gt_rois_ex[i * 32 + 1] != -1) {
					set_min_max(&gt_rois_ex[i * 32 + 1], 2, min_x, min_y, max_x, max_y);
				}
				if (gt_rois_ex[i * 32 + 16] != -1) {
					if (post_type_ == POST_V1) {
						set_min_max(&gt_rois_ex[i * 32 + 16], 4, min_x, min_y, max_x, max_y);
					}
					else if (post_type_ > POST_V1) {
						set_min_max(&gt_rois_ex[i * 32 + 16], 7, min_x, min_y, max_x, max_y);
					}
				}
			}
		}

		if (dc_rois && num_dcs) {
			for (int i = 0; i < num_dcs; i++) {
				set_min_max(&dc_rois[i * 5 + 1], 2, min_x, min_y, max_x, max_y);
			}
		}

		if (hn_rois && num_hns) {
			for (int i = 0; i < num_hns; i++) {
				set_min_max(&hn_rois[i * 5 + 1], 2, min_x, min_y, max_x, max_y);
			}
		}

		if (hp_rois && num_hps) {
			for (int i = 0; i < num_hps; i++) {
				set_min_max(&hp_rois[i * 5 + 1], 2, min_x, min_y, max_x, max_y);
			}
		}

		if (min_x == INT_MAX)
			min_x = width * 0.25;
		if (min_y == INT_MAX)
			min_y = height * 0.25;
		if (max_x == INT_MIN)
			max_x = width * 0.75;
		if (max_y == INT_MIN)
			max_y = height * 0.75;

		int min_w = max_x - min_x + 1;
		int min_h = max_y - min_y + 1;

		if (resized_width == 0 && resized_height == 0) {
			Dtype rnd_crop_ratio = 0.2;
			if (rnd_crop_) {
				int w = round(width * (1 - rnd_crop_ratio * ((Dtype)caffe_rng_rand() / UINT_MAX)));
				int h = round(height * (1 - rnd_crop_ratio * ((Dtype)caffe_rng_rand() / UINT_MAX)));
				crop_x = min<int>(min_x, round((width - w) * ((Dtype)caffe_rng_rand() / UINT_MAX)));
				crop_y = min<int>(min_y, round((height - h) * ((Dtype)caffe_rng_rand() / UINT_MAX)));
				crop_w = max<int>(max_x, crop_x + w - 1) - crop_x + 1;
				crop_h = max<int>(max_y, crop_y + h - 1) - crop_y + 1;
			}

			int im_size_min = min<int>(crop_w, crop_h);
			int im_size_max = max<int>(crop_w, crop_h);

			Dtype rnd_aspect_ratio = 0.2;
			Dtype im_scale = (rnd_scale_) ? (scale_min_ + caffe_rng_rand() % (scale_max_ - scale_min_)) / (Dtype)im_size_min : scale_min_ / (Dtype)im_size_min;
			if (round(im_scale * im_size_max) > scale_max_) {
				im_scale = scale_max_ / (Dtype)im_size_max;
			}
			im_scale_x = (rnd_aspect_) ? im_scale * ((1 + rnd_aspect_ratio) - 2 * rnd_aspect_ratio * ((Dtype)caffe_rng_rand() / UINT_MAX)) : im_scale;
			im_scale_y = (rnd_aspect_) ? im_scale * ((1 + rnd_aspect_ratio) - 2 * rnd_aspect_ratio * ((Dtype)caffe_rng_rand() / UINT_MAX)) : im_scale;
			im_scale_x = (Dtype)((int)(crop_w * im_scale_x / multiple_) * multiple_) / crop_w;
			im_scale_y = (Dtype)((int)(crop_h * im_scale_y / multiple_) * multiple_) / crop_h;

			resized_width = round(crop_w * im_scale_x);
			resized_height = round(crop_h * im_scale_y);
		}
		else {
			if (rnd_crop_) {
				int x1, y1, x2, y2;
				bool unconstrained_crop = unconstrained_rnd_crop_prob_ > 0 && (Dtype)caffe_rng_rand() / UINT_MAX < unconstrained_rnd_crop_prob_;
				if (!unconstrained_crop) {
					x1 = caffe_rng_rand() % (min_x + 1);
					y1 = caffe_rng_rand() % (min_y + 1);
					x2 = max_x + caffe_rng_rand() % (width - max_x);
					y2 = max_y + caffe_rng_rand() % (height - max_y);
				}
				else {
					min_x = width - 1;
					min_y = height - 1;
					max_x = 0;
					max_y = 0;
					min_w = width * unconstrained_rnd_crop_min_scale_;
					min_h = height * unconstrained_rnd_crop_min_scale_;
					x1 = caffe_rng_rand() % (width - min_w + 1);
					y1 = caffe_rng_rand() % (height - min_h + 1);
					x2 = x1 + min_w - 1 + caffe_rng_rand() % (width - (x1 + min_w) + 1);
					y2 = y1 + min_h - 1 + caffe_rng_rand() % (height - (y1 + min_h) + 1);
				}
				int w = x2 - x1 + 1;
				int h = y2 - y1 + 1;
				Dtype im_scale = (Dtype)resized_width / w;
				for (int i = 0; i < 10 && h > min_h && h < height; i++) {
					if (round(im_scale * h) > resized_height * 1.2) {
						Dtype dy = (h - resized_height / im_scale) / 2;
						y1 = min<int>(min_y, round(y1 + dy));
						y2 = max<int>(max_y, round(y2 - dy));
					}
					else if (round(im_scale * h) < resized_height * 0.8) {
						Dtype dy = (resized_height / im_scale - h) / 2;
						y1 = max<int>(0, round(y1 - dy));
						y2 = min<int>(height - 1, round(y2 + dy));
					}
					else {
						break;
					}
					h = y2 - y1 + 1;
				}
				im_scale = (Dtype)resized_height / h;
				for (int i = 0; i < 10 && w > min_w && w < width; i++) {
					if (round(im_scale * w) > resized_width * 1.2) {
						Dtype dx = (w - resized_width / im_scale) / 2;
						x1 = min<int>(min_x, round(x1 + dx));
						x2 = max<int>(max_x, round(x2 - dx));
					}
					else if (round(im_scale * w) < resized_width * 0.8) {
						Dtype dx = (resized_width / im_scale - w) / 2;
						x1 = max<int>(0, round(x1 - dx));
						x2 = min<int>(width - 1, round(x2 + dx));
					}
					else {
						break;
					}
					w = x2 - x1 + 1;
				}
				CHECK(x1 <= min_x && x1 >= 0 && y1 <= min_y && y1 >= 0 && x2 >= max_x && x2 <= width - 1 && y2 >= max_y && y2 <= height - 1)
					<< "x1=" << x1 << ", " << "y1=" << y1 << ", " << "x2=" << x2 << ", " << "y2=" << y2 << ", "
					<< "min_x=" << min_x << ", " << "max_x=" << max_x << ", " << "min_y=" << min_y << ", " << "max_y=" << max_y << ", " << "width=" << width << ", " << "height=" << height;
				crop_x = x1;
				crop_y = y1;
				crop_w = x2 - x1 + 1;
				crop_h = y2 - y1 + 1;
			}

			if (fixed_aspect_ > 0) {
				int crop_type = caffe_rng_rand() % 3;
				crop_w = round((float)height * fixed_aspect_);
				crop_h = height;
				if (crop_w > width) {
					crop_w = width;
					crop_h = round((float)width / fixed_aspect_);
					crop_x = 0;
					switch (crop_type) {
					case 0:
						crop_y = (height - crop_h) / 2;
						break;
					case 1:
						crop_y = 0;
						break;
					case 2:
						crop_y = height - crop_h;
						break;
					}
				}
				else {
					switch (crop_type) {
					case 0:
						crop_x = (width - crop_w) / 2;
						break;
					case 1:
						crop_x = 0;
						break;
					case 2:
						crop_x = width - crop_w;
						break;
					}
					crop_y = 0;
				}

				if (image_roi_.size() > 0) {
					crop_x = round(crop_x + crop_w * image_roi_[0]);
					crop_y = round(crop_y + crop_h * image_roi_[1]);
					crop_w = round(crop_w * (image_roi_[2] - image_roi_[0]));
					crop_h = round(crop_h * (image_roi_[3] - image_roi_[1]));
				}
			}

			if (use_closeness_) {

				int mxlen = (max_x - min_x);
				int mylen = (max_y - min_y);
				crop_w = (int)((Dtype)mxlen * 0.5 + 0.5 * ((Dtype)caffe_rng_rand() / UINT_MAX));
				crop_h = (int)((Dtype)resized_height * crop_w / (Dtype)resized_width);

				SetCropClose(crop_x, crop_w, (Dtype)min_x, (Dtype)max_x, 0.8);

				Dtype t = ((Dtype)caffe_rng_rand() / UINT_MAX);
				Dtype ratio = 0.8;
				crop_y = (int)(((Dtype)min_y - (1 - ratio) * (Dtype)crop_h) * t
					+ std::min((Dtype)max_y - (Dtype)crop_h, (Dtype)min_y) * (1 - t));

				crop_x = std::max(std::min((width - crop_w), crop_x), 0);
				crop_y = std::max(std::min((height - crop_h), crop_y), 0);

				closeness[0] = 0;
			}

			im_scale_x = (Dtype)resized_width / crop_w;
			im_scale_y = (Dtype)resized_height / crop_h;
		}

		if (use_closeness_) {
			int cnt_gts = 0;
			for (int i = 0; i < num_gts; i++) {
				gt_rois[i * 6 + 1] = (gt_rois[i * 6 + 1] - crop_x) * im_scale_x;
				gt_rois[i * 6 + 2] = (gt_rois[i * 6 + 2] - crop_y) * im_scale_y;
				gt_rois[i * 6 + 3] = (gt_rois[i * 6 + 3] - crop_x) * im_scale_x;
				gt_rois[i * 6 + 4] = (gt_rois[i * 6 + 4] - crop_y) * im_scale_y;
				if (gt_rois[i * 6 + 1] < resized_width &&  gt_rois[i * 6 + 3] - 1 > 0
					&& gt_rois[i * 6 + 2] < resized_height && gt_rois[i * 6 + 4] - 1 > 0) {
					gt_rois[cnt_gts * 6 + 1] = gt_rois[i * 6 + 1];
					gt_rois[cnt_gts * 6 + 2] = gt_rois[i * 6 + 2];
					gt_rois[cnt_gts * 6 + 3] = gt_rois[i * 6 + 3];
					gt_rois[cnt_gts * 6 + 4] = gt_rois[i * 6 + 4];


					if (gt_rois[cnt_gts * 6 + 1] < 0) {
						gt_rois[cnt_gts * 6 + 1] = 0;
					}
					if (gt_rois[cnt_gts * 6 + 2] < 0) {
						gt_rois[cnt_gts * 6 + 2] = 0;
					}

					if (gt_rois[cnt_gts * 6 + 3] > resized_width) {
						gt_rois[cnt_gts * 6 + 3] = resized_width;
					}

					if (gt_rois[cnt_gts * 6 + 4] > resized_height) {
						gt_rois[cnt_gts * 6 + 4] = resized_height;
						closeness[0] = 1;
					}
					cnt_gts++;
				}
				if (i > cnt_gts - 1) {
					gt_rois[i * 6 + 1] = -1;
					gt_rois[i * 6 + 2] = -1;
					gt_rois[i * 6 + 3] = -1;
					gt_rois[i * 6 + 4] = -1;
				}
			}

		}
		else if (gt_rois) {
			for (int i = 0; i < num_gts; i++) {
				transform_points(&gt_rois[i * 6 + 1], 2, crop_x, crop_y, im_scale_x, im_scale_y);
			}
			if (gt_rois_quad) {
				for (int i = 0; i < num_gts_quad; i++) {
					transform_points(&gt_rois_quad[i * 10 + 1], 4, crop_x, crop_y, im_scale_x, im_scale_y);
				}
			}
		}

		if (gt_rois_3d) {
			for (int i = 0; i < num_gts_3d; i++) {
				transform_points(&gt_rois_3d[i * 10 + 1], 4, crop_x, crop_y, im_scale_x, im_scale_y);
			}
		}

		if (gt_rois_new_3d) {
			for (int i = 0; i < num_gts_new_3d; i++) {
				transform_points(&gt_rois_new_3d[i * 17 + 1], 7, crop_x, crop_y, im_scale_x, im_scale_y);
			}
		}

		if (gt_rois_ex) {
			for (int i = 0; i < num_gts_ex; i++) {
				if (gt_rois_ex[i * 32 + 1] != -1) {
					transform_points(&gt_rois_ex[i * 32 + 1], 2, crop_x, crop_y, im_scale_x, im_scale_y);
				}
				if (gt_rois_ex[i * 32 + 16] != -1) {
					if (post_type_ == POST_V1) {
						transform_points(&gt_rois_ex[i * 32 + 16], 4, crop_x, crop_y, im_scale_x, im_scale_y);
					}
					else if (post_type_ > POST_V1) {
						transform_points(&gt_rois_ex[i * 32 + 16], 7, crop_x, crop_y, im_scale_x, im_scale_y);
					}
				}
			}
		}

		if (dc_rois) {
			for (int i = 0; i < num_dcs; i++) {
				transform_points(&dc_rois[i * 5 + 1], 2, crop_x, crop_y, im_scale_x, im_scale_y);
			}
		}

		if (hn_rois) {
			for (int i = 0; i < num_hns; i++) {
				transform_points(&hn_rois[i * 5 + 1], 2, crop_x, crop_y, im_scale_x, im_scale_y);
			}
		}

		if (hp_rois) {
			for (int i = 0; i < num_hps; i++) {
				transform_points(&hp_rois[i * 5 + 1], 2, crop_x, crop_y, im_scale_x, im_scale_y);
			}
		}
	}

	template<typename Dtype>
	void UDBDataLayer<Dtype>::SetCropClose(int& crop_start, const int crop_len, const Dtype gt_start, const Dtype gt_end, const Dtype ratio) {
		Dtype t = ((Dtype)caffe_rng_rand() / UINT_MAX);
		vector<int> list;

		if (gt_start < gt_end - crop_len) {
			list.push_back(0);
		}
		if (std::min(gt_end - crop_len, gt_start) > gt_start - (1 - ratio) * crop_len) {
			list.push_back(1);
		}
		if (gt_end - gt_start > ratio * crop_len
			&& gt_end - gt_start < crop_len) {
			list.push_back(2);
		}
		if (std::max(gt_start, gt_end - crop_len) < gt_end - ratio * crop_len) {
			list.push_back(3);
		}
		if (list.size()) {
			int sobj = list[caffe_rng_rand() % list.size()];



			if (sobj == 0) {
				crop_start = gt_start * t + (gt_end - crop_len) * (1 - t);
			}
			else if (sobj == 1) {
				crop_start = (gt_start - (1 - ratio) * crop_len) * t
					+ std::min(gt_end - crop_len, gt_start) * (1 - t);
			}
			else if (sobj == 2) {
				crop_start = (gt_end - crop_len) * t + gt_start * (1 - t);
			}
			else if (sobj == 3) {
				crop_start = std::max(gt_start, gt_end - crop_len) * t + (gt_end - ratio * crop_len) * (1 - t);
			}
		}
		else {
			crop_start = gt_start;
		}
	}

	template<typename Dtype>
	void UDBDataLayer<Dtype>::SetImgData(const unsigned char* src_data, Dtype* data,
		bool mirror, bool color_aug, int width, int height, int resized_width, int resized_height, int crop_x, int crop_y, Dtype im_scale_x, Dtype im_scale_y, const Dtype* affine_param) {
		const Dtype& a = affine_param[0];
		const Dtype& b = affine_param[1];
		const Dtype& c = affine_param[2];
		const Dtype& d = affine_param[3];
		const Dtype& ai = affine_param[4];
		const Dtype& bi = affine_param[5];
		const Dtype& ci = affine_param[6];
		const Dtype& di = affine_param[7];
		const int data_ch = input_channel_[input_type_];

		for (int i = 0; i < resized_height; i++) {
			for (int j = 0; j < resized_width; j++) {
				Dtype x = crop_x + j / im_scale_x;
				Dtype y = crop_y + i / im_scale_y;

				if (mirror) {
					x = width - x - 1;
				}

				if (rnd_affine_) {
					Dtype _x = ai * x + bi * y;
					Dtype _y = ci * x + di * y;
					x = _x;
					y = _y;
				}

				Dtype B = 0, G = 0, R = 0;

				if (x >= 0 && x <= width - 1 && y >= 0 && y <= height - 1) {
					int x0 = (int)x;
					int x1 = std::min(x0 + 1, width - 1);
					Dtype ax = x - x0;
					Dtype bx = 1 - ax;

					int y0 = (int)y;
					int y1 = std::min(y0 + 1, height - 1);
					Dtype ay = y - y0;
					Dtype by = 1 - ay;

					if (ax > 0 && ay > 0) {
						B += ax * ay * src_data[(y1 * width + x1) * 3 + 0];
						G += ax * ay * src_data[(y1 * width + x1) * 3 + 1];
						R += ax * ay * src_data[(y1 * width + x1) * 3 + 2];
					}
					if (ax > 0 && by > 0) {
						B += ax * by * src_data[(y0 * width + x1) * 3 + 0];
						G += ax * by * src_data[(y0 * width + x1) * 3 + 1];
						R += ax * by * src_data[(y0 * width + x1) * 3 + 2];
					}
					if (bx > 0 && ay > 0) {
						B += bx * ay * src_data[(y1 * width + x0) * 3 + 0];
						G += bx * ay * src_data[(y1 * width + x0) * 3 + 1];
						R += bx * ay * src_data[(y1 * width + x0) * 3 + 2];
					}
					if (bx > 0 && by > 0) {
						B += bx * by * src_data[(y0 * width + x0) * 3 + 0];
						G += bx * by * src_data[(y0 * width + x0) * 3 + 1];
						R += bx * by * src_data[(y0 * width + x0) * 3 + 2];
					}
				}
				if (use_tlr_blobReg_) {
					data[((0 * data_ch + 0) * resized_height + i) *  resized_width + j] = (B - mean0_) / 256.0;
					data[((0 * data_ch + 1) * resized_height + i) *  resized_width + j] = (G - mean1_) / 256.0;
					data[((0 * data_ch + 2) * resized_height + i) *  resized_width + j] = (R - mean2_) / 256.0;
				}
				else {
					data[((0 * data_ch + 0) * resized_height + i) *  resized_width + j] = B - mean0_ - mean0_noise_;
					data[((0 * data_ch + 1) * resized_height + i) *  resized_width + j] = G - mean1_ - mean1_noise_;
					data[((0 * data_ch + 2) * resized_height + i) *  resized_width + j] = R - mean2_ - mean2_noise_;
				}
			}
		}
		if (color_aug) {
			cv::Mat covData(3, 3, CV_32FC1, cv::Scalar::all(0));
			Dtype* covptr = covData.ptr<Dtype>();
			Dtype t1 = 0, t2 = 0, t3 = 0;

			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					t1 = data[((0 * data_ch + 0) * resized_height + i) * resized_width + j] / 255;
					t2 = data[((0 * data_ch + 1) * resized_height + i) * resized_width + j] / 255;
					t3 = data[((0 * data_ch + 2) * resized_height + i) * resized_width + j] / 255;

					covptr[0] = covptr[0] + t1 * t1;
					covptr[1] = covptr[1] + t1 * t2;
					covptr[2] = covptr[2] + t1 * t3;
					covptr[3] = covptr[3] + t2 * t1;
					covptr[4] = covptr[4] + t2 * t2;
					covptr[5] = covptr[5] + t2 * t3;
					covptr[6] = covptr[6] + t3 * t1;
					covptr[7] = covptr[7] + t3 * t2;
					covptr[8] = covptr[8] + t3 * t3;
				}
			}

			// Eigen values and vectors
			cv::Mat eig_val, eig_vec;
			eigen(covData, eig_val, eig_vec);
			//cout << "Eigenvalues" << endl << eig_val << endl;
			//cout << "Eigenvectors" << endl << eig_vec << endl;

			eig_val.at< Dtype>(0, 0) = 0.1 * sqrt(eig_val.at< Dtype>(0, 0)) * (Dtype)caffe_rng_rand() / UINT_MAX;
			eig_val.at< Dtype>(1, 0) = 0.1 * sqrt(eig_val.at< Dtype>(1, 0)) * (Dtype)caffe_rng_rand() / UINT_MAX;
			eig_val.at< Dtype>(2, 0) = 0.1 * sqrt(eig_val.at< Dtype>(2, 0)) * (Dtype)caffe_rng_rand() / UINT_MAX;

			if (isnan(eig_val.at<Dtype>(0, 0)))
				eig_val.at<Dtype>(0, 0) = 0;
			if (isnan(eig_val.at<Dtype>(1, 0)))
				eig_val.at<Dtype>(1, 0) = 0;
			if (isnan(eig_val.at<Dtype>(2, 0)))
				eig_val.at<Dtype>(2, 0) = 0;

			cv::Mat color_val(3, 1, CV_32FC1, cv::Scalar::all(0));
			color_val = eig_vec * eig_val;

			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					data[((0 * data_ch + 0) * resized_height + i) *  resized_width + j] = data[((0 * data_ch + 0) * resized_height + i) *  resized_width + j] + color_val.at<Dtype>(0, 0);
					data[((0 * data_ch + 1) * resized_height + i) *  resized_width + j] = data[((0 * data_ch + 1) * resized_height + i) *  resized_width + j] + color_val.at<Dtype>(1, 0);
					data[((0 * data_ch + 2) * resized_height + i) *  resized_width + j] = data[((0 * data_ch + 2) * resized_height + i) *  resized_width + j] + color_val.at<Dtype>(2, 0);
				}
			}
		}
		if (use_limit_) {
			const int num_pixels = resized_height * resized_width * data_ch;
			for (int i = 0; i < num_pixels; i++) {
				data[i] = (data[i] > max_limit_) ? max_limit_ : ((data[i] < min_limit_) ? min_limit_ : data[i]);
			}
		}
		if (input_type_ == 1) { // RGB + Gray
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					const Dtype B = data[((0 * data_ch + 0) * resized_height + i) * resized_width + j] + mean0_;
					const Dtype G = data[((0 * data_ch + 1) * resized_height + i) * resized_width + j] + mean1_;
					const Dtype R = data[((0 * data_ch + 2) * resized_height + i) * resized_width + j] + mean2_;
					data[((0 * data_ch + 3) * resized_height + i) * resized_width + j] = 0.299 * R + 0.587 * G + 0.114 * B;
				}
			}
		}
		else if (input_type_ == 2) { // Gray + Laplacian + SobelX + SobelY
			cv::Mat buf1(resized_height, resized_width, CV_8UC1);
			cv::Mat buf2(resized_height, resized_width, CV_8UC1);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					const Dtype B = data[((0 * data_ch + 0) * resized_height + i) * resized_width + j] + mean0_;
					const Dtype G = data[((0 * data_ch + 1) * resized_height + i) * resized_width + j] + mean1_;
					const Dtype R = data[((0 * data_ch + 2) * resized_height + i) * resized_width + j] + mean2_;
					buf1.data[i * resized_width + j] = 0.299 * R + 0.587 * G + 0.114 * B;
					data[((0 * data_ch + 0) * resized_height + i) * resized_width + j] = buf1.data[i * resized_width + j];
				}
			}
			cv::Laplacian(buf1, buf2, CV_8UC1, 3);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					data[((0 * data_ch + 1) * resized_height + i) * resized_width + j] = buf2.data[i * resized_width + j];
				}
			}
			cv::Sobel(buf1, buf2, CV_8UC1, 1, 0, 3);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					data[((0 * data_ch + 2) * resized_height + i) * resized_width + j] = buf2.data[i * resized_width + j];
				}
			}
			cv::Sobel(buf1, buf2, CV_8UC1, 0, 1, 3);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					data[((0 * data_ch + 3) * resized_height + i) * resized_width + j] = buf2.data[i * resized_width + j];
				}
			}
		}
		else if (input_type_ == 3) { // Gray + Laplacian + Canny
			cv::Mat buf1(resized_height, resized_width, CV_8UC1);
			cv::Mat buf2(resized_height, resized_width, CV_8UC1);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					const Dtype B = data[((0 * data_ch + 0) * resized_height + i) * resized_width + j] + mean0_;
					const Dtype G = data[((0 * data_ch + 1) * resized_height + i) * resized_width + j] + mean1_;
					const Dtype R = data[((0 * data_ch + 2) * resized_height + i) * resized_width + j] + mean2_;
					buf1.data[i * resized_width + j] = 0.299 * R + 0.587 * G + 0.114 * B;
					data[((0 * data_ch + 0) * resized_height + i) * resized_width + j] = buf1.data[i * resized_width + j];
				}
			}
			cv::Laplacian(buf1, buf2, CV_8UC1, 3);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					data[((0 * data_ch + 1) * resized_height + i) * resized_width + j] = buf2.data[i * resized_width + j];
				}
			}
			cv::Canny(buf1, buf2, 100, 127);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					data[((0 * data_ch + 2) * resized_height + i) * resized_width + j] = buf2.data[i * resized_width + j];
				}
			}
		}
		else if (input_type_ == 4) { // Gray + Laplacian + SobelX + SobelY + Canny
			cv::Mat buf1(resized_height, resized_width, CV_8UC1);
			cv::Mat buf2(resized_height, resized_width, CV_8UC1);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					const Dtype B = data[((0 * data_ch + 0) * resized_height + i) * resized_width + j] + mean0_;
					const Dtype G = data[((0 * data_ch + 1) * resized_height + i) * resized_width + j] + mean1_;
					const Dtype R = data[((0 * data_ch + 2) * resized_height + i) * resized_width + j] + mean2_;
					buf1.data[i * resized_width + j] = 0.299 * R + 0.587 * G + 0.114 * B;
					data[((0 * data_ch + 0) * resized_height + i) * resized_width + j] = buf1.data[i * resized_width + j];
				}
			}
			cv::Laplacian(buf1, buf2, CV_8UC1, 3);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					data[((0 * data_ch + 1) * resized_height + i) * resized_width + j] = buf2.data[i * resized_width + j];
				}
			}
			cv::Sobel(buf1, buf2, CV_8UC1, 1, 0, 3);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					data[((0 * data_ch + 2) * resized_height + i) * resized_width + j] = buf2.data[i * resized_width + j];
				}
			}
			cv::Sobel(buf1, buf2, CV_8UC1, 0, 1, 3);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					data[((0 * data_ch + 3) * resized_height + i) * resized_width + j] = buf2.data[i * resized_width + j];
				}
			}
			cv::Canny(buf1, buf2, 100, 127);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					data[((0 * data_ch + 4) * resized_height + i) * resized_width + j] = buf2.data[i * resized_width + j];
				}
			}
		}
		else if (input_type_ == 5) { // RGB + Laplacian
			cv::Mat buf1(resized_height, resized_width, CV_8UC1);
			cv::Mat buf2(resized_height, resized_width, CV_8UC1);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					const Dtype B = data[((0 * data_ch + 0) * resized_height + i) * resized_width + j] + mean0_;
					const Dtype G = data[((0 * data_ch + 1) * resized_height + i) * resized_width + j] + mean1_;
					const Dtype R = data[((0 * data_ch + 2) * resized_height + i) * resized_width + j] + mean2_;
					buf1.data[i * resized_width + j] = 0.299 * R + 0.587 * G + 0.114 * B;
				}
			}
			cv::Laplacian(buf1, buf2, CV_8UC1, 3);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					data[((0 * data_ch + 3) * resized_height + i) * resized_width + j] = buf2.data[i * resized_width + j];
				}
			}
		}
		else if (input_type_ == 6) { // RGB + SobelX + SobelY
			cv::Mat buf1(resized_height, resized_width, CV_8UC1);
			cv::Mat buf2(resized_height, resized_width, CV_8UC1);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					const Dtype B = data[((0 * data_ch + 0) * resized_height + i) * resized_width + j] + mean0_;
					const Dtype G = data[((0 * data_ch + 1) * resized_height + i) * resized_width + j] + mean1_;
					const Dtype R = data[((0 * data_ch + 2) * resized_height + i) * resized_width + j] + mean2_;
					buf1.data[i * resized_width + j] = 0.299 * R + 0.587 * G + 0.114 * B;
				}
			}
			cv::Sobel(buf1, buf2, CV_8UC1, 1, 0, 3);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					data[((0 * data_ch + 3) * resized_height + i) * resized_width + j] = buf2.data[i * resized_width + j];
				}
			}
			cv::Sobel(buf1, buf2, CV_8UC1, 0, 1, 3);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					data[((0 * data_ch + 4) * resized_height + i) * resized_width + j] = buf2.data[i * resized_width + j];
				}
			}
		}
		else if (input_type_ == 7) { // RGB + Laplacian + SobelX + SobelY
			cv::Mat buf1(resized_height, resized_width, CV_8UC1);
			cv::Mat buf2(resized_height, resized_width, CV_8UC1);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					const Dtype B = data[((0 * data_ch + 0) * resized_height + i) * resized_width + j] + mean0_;
					const Dtype G = data[((0 * data_ch + 1) * resized_height + i) * resized_width + j] + mean1_;
					const Dtype R = data[((0 * data_ch + 2) * resized_height + i) * resized_width + j] + mean2_;
					buf1.data[i * resized_width + j] = 0.299 * R + 0.587 * G + 0.114 * B;
				}
			}
			cv::Laplacian(buf1, buf2, CV_8UC1, 3);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					data[((0 * data_ch + 3) * resized_height + i) * resized_width + j] = buf2.data[i * resized_width + j];
				}
			}
			cv::Sobel(buf1, buf2, CV_8UC1, 1, 0, 3);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					data[((0 * data_ch + 4) * resized_height + i) * resized_width + j] = buf2.data[i * resized_width + j];
				}
			}
			cv::Sobel(buf1, buf2, CV_8UC1, 0, 1, 3);
			for (int i = 0; i < resized_height; i++) {
				for (int j = 0; j < resized_width; j++) {
					data[((0 * data_ch + 5) * resized_height + i) * resized_width + j] = buf2.data[i * resized_width + j];
				}
			}
		}
	}

	template<typename Dtype>
	void UDBDataLayer<Dtype>::SetSegData(const unsigned char* src_data, Dtype* data, int seg_index,
		bool mirror, int width, int height, int resized_width, int resized_height, int crop_x, int crop_y, Dtype im_scale_x, Dtype im_scale_y, const Dtype* affine_param) {
		const Dtype& a = affine_param[0];
		const Dtype& b = affine_param[1];
		const Dtype& c = affine_param[2];
		const Dtype& d = affine_param[3];
		const Dtype& ai = affine_param[4];
		const Dtype& bi = affine_param[5];
		const Dtype& ci = affine_param[6];
		const Dtype& di = affine_param[7];

		for (int i = 0; i < resized_height; i++) {
			for (int j = 0; j < resized_width; j++) {
				Dtype x = crop_x + j / im_scale_x;
				Dtype y = crop_y + i / im_scale_y;

				if (mirror) {
					x = width - x - 1;
				}

				if (rnd_affine_) {
					Dtype _x = ai * x + bi * y;
					Dtype _y = ci * x + di * y;
					x = _x;
					y = _y;
				}
				int label = 0;
				if (x >= 0 && x <= width - 1 && y >= 0 && y <= height - 1) {
					int x0 = (int)x;
					int y0 = (int)y;
					int segColor = RGB2INT(src_data[(y0 * width + x0) * 3 + 2],
						src_data[(y0 * width + x0) * 3 + 1],
						src_data[(y0 * width + x0) * 3 + 0]);
					if (seg_output_rgb2int_) {
						label = segColor;
					}
					else {
						auto it = segment_map_[seg_index].find(segColor);
						CHECK(it != segment_map_[seg_index].end()) << "Unknown Color - R: " << (int)src_data[(y0 * width + x0) * 3 + 2] << ", G: " << (int)src_data[(y0 * width + x0) * 3 + 1] << ", B: " << (int)src_data[(y0 * width + x0) * 3 + 0];
						label = it->second;  //seg
					}
				}
				int data_c = (i % segment_map_scale_[seg_index]) * segment_map_scale_[seg_index] + (j % segment_map_scale_[seg_index]);
				int data_y = i / segment_map_scale_[seg_index];
				int data_x = j / segment_map_scale_[seg_index];
				if (!segment_label_resize_[seg_index]) {
					data[(data_c * (resized_height / segment_map_scale_[seg_index]) + data_y) * (resized_width / segment_map_scale_[seg_index]) + data_x] = label;
				}
				else if (data_c == 0) {
					data[data_y * (resized_width / segment_map_scale_[seg_index]) + data_x] = label;
				}
			}
		}
	}

	template<typename Dtype>
	void UDBDataLayer<Dtype>::MakeEdge(const cv::Mat& src_data, Dtype* _edge, bool mirror, int width, int height,
		int resized_width, int resized_height, int crop_x, int crop_y, Dtype im_scale_x, Dtype im_scale_y) {

		cv::Mat label = src_data.clone();
		cv::Mat gray, reImg, laplacian, edge;
		int kernel_size = 3;
		int scale = 1;
		int delta = 0;
		int ddepth = CV_16S;

		for (int y = 0; y < height; ++y) {
			const cv::Vec3b* ptImg = src_data.ptr<cv::Vec3b>(y);
			cv::Vec3b* ptLabel = label.ptr<cv::Vec3b>(y);
			for (int x = 0; x < width; ++x) {
				if ((ptImg[x][0] == 128 && ptImg[x][1] == 64 && ptImg[x][2] == 128) ||
					(ptImg[x][0] == 255 && ptImg[x][1] == 255 && ptImg[x][2] == 0) ||
					(ptImg[x][0] == 0 && ptImg[x][1] == 255 && ptImg[x][2] == 255)) {
					ptLabel[x][0] = 255;
					ptLabel[x][1] = 255;
					ptLabel[x][2] = 255;

				}
				else {
					ptLabel[x][0] = 0;
					ptLabel[x][1] = 0;
					ptLabel[x][2] = 0;
				}
			}
		}

		/// Convert the image to grayscale
		cvtColor(label, gray, CV_BGR2GRAY);
		// Crop the image with ROI
		cv::Rect rect(crop_x, crop_y, width - crop_x, height - crop_y);
		reImg = gray(rect);
		// Resize the image
		cv::resize(reImg, reImg, cv::Size(resized_width / segment_map_scale_[0], resized_height / segment_map_scale_[0]), 0, 0, CV_INTER_NN);
		if (mirror) {
			cv::flip(reImg, reImg, 1);

		}
		// Laplacian Edge
		Laplacian(reImg, laplacian, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT);
		convertScaleAbs(laplacian, edge);

		for (int y = 0; y < edge.rows; ++y) {
			for (int x = 0; x < edge.cols; ++x) {
				if (edge.at<uchar>(y, x) == 0) {
					_edge[y * edge.cols + x] = 0;
				}
				else if (edge.at<uchar>(y, x) == 255) {
					_edge[y * edge.cols + x] = 1;
				}
				else {
					_edge[y * edge.cols + x] = 0;
				}
			}
		}
	}

	template<typename Dtype>
	void UDBDataLayer<Dtype>::putGaussianMaps(Dtype* entry, cv::Point2f center, int stride, int grid_x, int grid_y, float sigma) {
		//LOG(INFO) << "putGaussianMaps here we start for " << center.x << " " << center.y;
		float start = stride / 2.0 - 0.5; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
		for (int g_y = 0; g_y < grid_y; g_y++) {
			for (int g_x = 0; g_x < grid_x; g_x++) {
				float x = start + g_x * stride;
				float y = start + g_y * stride;
				float d2 = (x - center.x) * (x - center.x) + (y - center.y) * (y - center.y);
				float exponent = d2 / 2.0 / sigma / sigma;
				if (exponent > 4.6052) { //ln(100) = -ln(1%)
					continue;
				}
				entry[g_y * grid_x + g_x] += exp(-exponent);
				if (entry[g_y * grid_x + g_x] > 1)
					entry[g_y * grid_x + g_x] = 1;
			}
		}
	}

	template<typename Dtype>
	void UDBDataLayer<Dtype>::SetTLRRegData(const cv::Mat& img, const std::vector<cv::Point2i>& ct_pt, Dtype* data, bool mirror,
		int width, int height, int resized_width, int resized_height, int crop_x, int crop_y, Dtype im_scale_x, Dtype im_scale_y) {

		int stride = 1;
		int rezX = resized_width;
		int rezY = resized_height;
		int grid_x = rezX / stride;
		int grid_y = rezY / stride;
		int channelOffset = grid_y * grid_x;

		// label resize
		int sigma = 5;

		for (int i = 0; i < 5; ++i) {
			cv::Point2i center = ct_pt[i];
			if (mirror) {
				center.x = grid_x - center.x - 1;
			}
			if (!(center.x == 0 && center.y == 0))
				putGaussianMaps(data + i * channelOffset, center, stride, grid_x, grid_y, sigma);
		}

		//put background channel
		for (int g_y = 0; g_y < grid_y; g_y++) {
			for (int g_x = 0; g_x < grid_x; g_x++) {
				float maximum = 0;
				//second background channel
				for (int i = 0; i < 5; ++i) {
					maximum = (maximum > data[i * channelOffset + g_y * grid_x + g_x]) ? maximum : data[i * channelOffset + g_y * grid_x + g_x];
				}
				data[5 * channelOffset + g_y * grid_x + g_x] = max(1.0 - maximum, 0.0);
			}
		}

		// display the results
		//for (int i = 0; i < 6; i++) {
		//  cv::Mat draw = cv::Mat::zeros(grid_y, grid_x, CV_8UC1);
		//  for (int g_y = 0; g_y < grid_y; g_y++) {
		//    for (int g_x = 0; g_x < grid_x; g_x++) {
		//      draw.at<uchar>(g_y, g_x) = (int)(data[i * channelOffset + g_y * grid_x + g_x] * 255);
		//      //printf("%f ", transformed_label_entry[g_y*grid_x + g_x]*255);
		//    }
		//  }
		//  applyColorMap(draw, draw, cv::COLORMAP_JET);
		//  addWeighted(draw, 0.5, img, 0.5, 0.0, draw);
		//  char imagename[100];
		//  sprintf(imagename, "output/augment_%04d_label_part_%02d.jpg", 0, i);
		//  imwrite(imagename, draw);
		//}
	}

	template<typename Dtype>
	bool UDBDataLayer<Dtype>::select_space_randomly(cv::Mat& mask_map, const int patch_width, const int patch_height, const int min_x, const int min_y, const int max_x, const int max_y, int& random_left_x, int& random_top_y, int& available_width, int& available_height) {
		float patch_ratio = (float)patch_width / patch_height;
		int available_min_width = patch_width * fp_patch_min_ratio_;
		int available_max_width = patch_width * fp_patch_max_ratio_;
		int available_min_height = patch_height * fp_patch_min_ratio_;
		int available_max_height = patch_height * fp_patch_max_ratio_;

		int start_x, start_y, acc_sum;
		float current_ratio, prev_ratio;
		std::vector<int> directions{ -1, 1 };
		int x_direction = 0;
		int y_direction = 0;
		int turn = 1;
		int try_count = 0;

		while (1) {
			try_count += 1;
			start_x = rand() % (max_x - min_x + 1) + min_x;
			start_y = rand() % (max_y - min_y + 1) + min_y;
			std::vector<int> available_x{ start_x, start_x };
			std::vector<int> available_y{ start_y, start_y };
			if (mask_map.at<uchar>(start_y, start_x) != 0)
				continue;

			current_ratio = 1;
			std::vector<int> empty_space_x{ 1, 1 };
			std::vector<int> empty_space_y{ 1, 1 };

			while (1) {
				if (turn == 1 && !(empty_space_x[0] == -1 && empty_space_x[1] == -1)) {
					available_x[x_direction] += directions[x_direction];
					acc_sum = ((min_x < available_x[0] && available_x[1] < max_x) && available_width < available_max_width) ? cv::sum(cv::Mat(mask_map, cv::Rect(available_x[x_direction], available_y[0], 1, available_y[1] - available_y[0] + 1)))[0] : 1;

					if (acc_sum != 0) {
						available_x[x_direction] -= directions[x_direction];
						empty_space_x[x_direction] = -1;
					}
					x_direction = 1 - x_direction;
					available_width = available_x[1] - available_x[0];
				}
				else if (turn == -1 && !(empty_space_y[0] == -1 && empty_space_y[1] == -1)) {
					available_y[y_direction] += directions[y_direction];
					acc_sum = (min_y < available_y[0] && available_y[1] < max_y && available_height < available_max_height) ? cv::sum(cv::Mat(mask_map, cv::Rect(available_x[0], available_y[y_direction], available_x[1] - available_x[0] + 1, 1)))[0] : 1;

					if (acc_sum != 0) {
						available_y[y_direction] -= directions[y_direction];
						empty_space_y[y_direction] = -1;
					}
					y_direction = 1 - y_direction;
					available_height = available_y[1] - available_y[0];
				}
				else
					break;

				prev_ratio = current_ratio;
				current_ratio = float(available_x[1] - available_x[0] + 1) / (available_y[1] - available_y[0] + 1);
				if (std::abs(patch_ratio - current_ratio) > std::abs(patch_ratio - prev_ratio))
					turn *= -1;
			}

			if (available_width >= available_min_width && available_height >= available_min_height) {
				random_left_x = available_x[0];
				random_top_y = available_y[0];
				break;
			}
			if (try_count >= 10)
				return false;
		}
		//random resize
		int rnd_resized_available_width, rnd_resized_available_height;
		if (current_ratio < 1) {
			rnd_resized_available_height = rand() % (available_height - available_min_height + 1) + available_min_height;
			rnd_resized_available_width = available_width * ((float)rnd_resized_available_height / available_height);
		}
		else {
			rnd_resized_available_width = rand() % (available_width - available_min_width + 1) + available_min_width;
			rnd_resized_available_height = available_height * ((float)rnd_resized_available_width / available_width);
		}
		CHECK((rnd_resized_available_width >= available_min_width) || (rnd_resized_available_height >= available_min_height));
		available_width = rnd_resized_available_width;
		available_height = rnd_resized_available_height;
		return true;
	}

	template<typename Dtype>
	void UDBDataLayer<Dtype>::crop_image_randomly(const int left_x, const int top_y, const int right_x, const int bottom_y, db::UDBDatum* datum, cv::Mat& new_img, std::vector<db::Box2D>& new_box2d, std::vector<db::ObjectGT>& new_object_gt) {
		int image_width = datum->img_.cols;
		int image_height = datum->img_.rows;
		float image_ratio = (float)image_width / image_height;

		int window_width = right_x - left_x + 1;
		int window_height = bottom_y - top_y + 1;
		int crop_width;
		int crop_height;

		int crop_min_width = std::min((int)(window_width * mosaic_crop_scale_min_), image_width);
		int crop_max_width = std::min((int)(window_width * mosaic_crop_scale_max_), image_width);
		int crop_min_height = std::min((int)(window_height * mosaic_crop_scale_min_), image_height);
		int crop_max_height = std::min((int)(window_height * mosaic_crop_scale_max_), image_height);
		if ((float)crop_max_width / window_width > (float)crop_max_height / window_height) {
			crop_height = rand() % (crop_max_height - crop_min_height + 1) + crop_min_height;
			crop_width = (float)crop_height / window_height * window_width;
		}
		else {
			crop_width = rand() % (crop_max_width - crop_min_width + 1) + crop_min_width;
			crop_height = (float)crop_width / window_width * window_height;
		}

		int crop_start_x = crop_start_x = rand() % (image_width - crop_width + 1);
		int crop_start_y = crop_start_y = rand() % (image_height - crop_height + 1);
		cv::Mat img_roi = new_img(cv::Rect(cv::Point(left_x, top_y), cv::Size(window_width, window_height)));
		cv::Mat crop_roi = datum->img_(cv::Rect(crop_start_x, crop_start_y, crop_width, crop_height)).clone();
		resize(crop_roi, crop_roi, cv::Size(window_width, window_height), 0, 0, cv::INTER_CUBIC);
		float resize_scale = (float)window_width / crop_width;

		int mosaic_flip = rand() % 2;
		//mosaic_flip = 0;
		for (int i = 0; i < datum->box_2d_.size(); i++) {
			db::Box2D box2d_ = datum->box_2d_[i];
			int intersection_width = std::min((int)box2d_.x2, crop_start_x + crop_width - 1) - std::max((int)box2d_.x1, crop_start_x);
			int intersection_height = std::min((int)box2d_.y2, crop_start_y + crop_height - 1) - std::max((int)box2d_.y1, crop_start_y);
			if (intersection_width > 0 && intersection_height > 0) {
				float ioa = (float)(intersection_width * intersection_height) / ((box2d_.x2 - box2d_.x1 + 1) * (box2d_.y2 - box2d_.y1 + 1));
				int x1_new = std::max(std::min((int)((box2d_.x1 - crop_start_x) * resize_scale + left_x), right_x), left_x);
				int x2_new = std::min((int)std::ceil((box2d_.x2 - crop_start_x) * resize_scale + left_x), right_x);
				int y1_new = std::max(std::min((int)((box2d_.y1 - crop_start_y) * resize_scale + top_y), bottom_y), top_y);
				int y2_new = std::min((int)std::ceil((box2d_.y2 - crop_start_y) * resize_scale + top_y), bottom_y);
				CHECK(x1_new <= x2_new && y1_new <= y2_new) << " x1 " << x1_new << " x2 " << x2_new << " y1 " << y1_new << " y2 " << y2_new;
				box2d_.x1 = (mosaic_flip) ? (right_x - x2_new + left_x) : x1_new;
				box2d_.x2 = (mosaic_flip) ? (right_x - x1_new + left_x) : x2_new;
				box2d_.y1 = y1_new;
				box2d_.y2 = y2_new;

				if (ioa < mosaic_ioa_threshold_)
					box2d_.label = -100000;

				db::ObjectGT new_obj_gt;
				new_obj_gt.box_2d_idx_ = new_box2d.size();
				new_object_gt.push_back(new_obj_gt);
				new_box2d.push_back(box2d_);
			}
		}

		if (mosaic_flip)
			cv::flip(crop_roi, crop_roi, 1);
		crop_roi.copyTo(img_roi);
	}

	template<typename Dtype>
	bool UDBDataLayer<Dtype>::maintain_aspect_ratio(db::UDBDatum* datum, cv::Mat& new_img, std::vector<db::Box2D>& new_box2d, std::vector<db::ObjectGT>& new_object_gt, int aspect_width, int aspect_height, float ioa_threshold) {
		int img_width = datum->img_.cols;
		int img_height = datum->img_.rows;
		bool valid_check = false;
		if (img_width * aspect_height == img_height * aspect_width)
			return false;
		std::vector<int> valid_objects;
		for (int i = 0; i < datum->box_2d_.size(); i++) {
			if (datum->box_2d_[i].label > 0) {
				valid_objects.push_back(i);
			}
		}
		int use_object_index = valid_objects[caffe_rng_rand() % valid_objects.size()];

		db::Box2D selected_obj = datum->box_2d_[use_object_index];
		if (img_width * aspect_height > img_height * aspect_width) {
			int obj_min_x1 = INT_MAX;
			int obj_min_x2 = 0;
			int obj_max_x1 = 0;
			int obj_max_x2 = INT_MIN;
			int adjusted_width = (int)(img_height * aspect_width / aspect_height);
			int adjusted_x1, adjusted_x2;
			if ((selected_obj.x2 - selected_obj.x1 + 1) >= adjusted_width) {
				adjusted_x1 = (caffe_rng_rand() % ((int)selected_obj.x2 - (int)selected_obj.x1 + 1 - adjusted_width + 1)) + (int)selected_obj.x1;
				adjusted_x2 = adjusted_x1 + adjusted_width - 1;
			}
			else {
				int min_start = std::max((int)selected_obj.x2 - adjusted_width + 1, 0);
				int max_start = std::min((int)selected_obj.x1, (int)img_width - adjusted_width);
				CHECK(min_start <= max_start) << " min_start " << min_start << " max_start " << max_start;
				adjusted_x1 = (caffe_rng_rand() % (max_start - min_start + 1)) + min_start;
				adjusted_x2 = adjusted_x1 + adjusted_width - 1;
			}
			new_img = datum->img_(cv::Rect(adjusted_x1, 0, adjusted_width, img_height)).clone();
			for (int i = 0; i < datum->box_2d_.size(); i++) {
				db::Box2D box2d_ = datum->box_2d_[i];
				int intersection_width = std::min((int)box2d_.x2, adjusted_x2) - std::max((int)box2d_.x1, adjusted_x1) + 1;
				if (intersection_width > 0) {
					float ioa = (float)(intersection_width) / (box2d_.x2 - box2d_.x1 + 1);
					int x1_new = std::min(std::max((int)box2d_.x1 - adjusted_x1, 0), adjusted_width - 1);
					int x2_new = std::max(std::min((int)box2d_.x2 - adjusted_x1, adjusted_width - 1), 0);
					CHECK(x1_new <= x2_new) << " x1 " << x1_new << " x2 " << x2_new;
					box2d_.x1 = x1_new;
					box2d_.x2 = x2_new;
					if (ioa < ioa_threshold && i != use_object_index)
						box2d_.label = -100000;
					else if (box2d_.label > 0) {
						valid_check = true;
					}
					db::ObjectGT new_obj_gt;
					new_obj_gt.box_2d_idx_ = new_box2d.size();
					new_object_gt.push_back(new_obj_gt);
					new_box2d.push_back(box2d_);
				}
			}
		}
		else if (img_height * aspect_width > img_width * aspect_height) {
			int obj_min_y1 = INT_MAX;
			int obj_min_y2 = 0;
			int obj_max_y1 = 0;
			int obj_max_y2 = INT_MIN;
			int adjusted_height = (int)(img_width * aspect_height / aspect_width);
			int adjusted_y1, adjusted_y2;
			if ((selected_obj.y2 - selected_obj.y1 + 1) >= adjusted_height) {
				adjusted_y1 = (caffe_rng_rand() % ((int)selected_obj.y2 - (int)selected_obj.y1 + 1 - adjusted_height + 1)) + (int)selected_obj.y1;
				adjusted_y2 = adjusted_y1 + adjusted_height - 1;
			}
			else {
				int min_start = std::max((int)selected_obj.y2 - adjusted_height + 1, 0);
				int max_start = std::min((int)selected_obj.y1, (int)img_height - adjusted_height);
				CHECK(min_start <= max_start) << " min_start " << min_start << " max_start " << max_start;
				adjusted_y1 = (caffe_rng_rand() % (max_start - min_start + 1)) + min_start;
				adjusted_y2 = adjusted_y1 + adjusted_height - 1;
			}
			new_img = datum->img_(cv::Rect(0, adjusted_y1, img_width, adjusted_height)).clone();
			for (int i = 0; i < datum->box_2d_.size(); i++) {
				db::Box2D box2d_ = datum->box_2d_[i];
				int intersection_height = std::min((int)box2d_.y2, adjusted_y2) - std::max((int)box2d_.y1, adjusted_y1) + 1;
				if (intersection_height > 0) {
					float ioa = (float)(intersection_height) / (box2d_.y2 - box2d_.y1 + 1);
					int y1_new = std::min(std::max((int)box2d_.y1 - adjusted_y1, 0), adjusted_height - 1);
					int y2_new = std::max(std::min((int)box2d_.y2 - adjusted_y1, adjusted_height - 1), 0);
					CHECK(y1_new <= y2_new) << " y1 " << y1_new << " y2 " << y2_new;
					box2d_.y1 = y1_new;
					box2d_.y2 = y2_new;
					if (ioa < ioa_threshold && i != use_object_index)
						box2d_.label = -100000;
					else if (box2d_.label > 0) {
						valid_check = true;
					}
					db::ObjectGT new_obj_gt;
					new_obj_gt.box_2d_idx_ = new_box2d.size();
					new_object_gt.push_back(new_obj_gt);
					new_box2d.push_back(box2d_);
				}
			}
		}
		return true;
	}

	template<typename Dtype>
	bool UDBDataLayer<Dtype>::TransformData(vector<db::UDBDatum*> data, UDBBatch<Dtype>* batch) {
		// find variables to set
		bool present_img = false, present_seg = false, present_scene_lbl = false, present_failsafe = false, present_meta_info = false;

		for (int i = 0; i < data.size(); i++) {
			if (!data[i]->img_.empty())
				present_img = true;
			if (!data[i]->seg_.empty())
				present_seg = true;
			if (data[i]->scene_lbl_.empty())
				present_scene_lbl = true;
			if (data[i]->failsafe_ > -3)
				present_failsafe = true;
			if (data[i]->meta_info_.empty())
				present_meta_info = true;
		}
		int num_all_gts = 0, gt_roi_index = 0;
		int num_all_gts_quad = 0, gt_roi_quad_index = 0;
		int num_all_gts_3d = 0, gt_roi_3d_index = 0;
		int num_all_gts_new_3d = 0, gt_roi_new_3d_index = 0;
		int num_all_gts_ex = 0, gt_roi_ex_index = 0;
		int num_all_dcs = 0, dc_roi_index = 0;
		int num_all_hns = 0, hn_roi_index = 0;
		int num_all_hps = 0, hp_roi_index = 0;
		int lane_type_label_data_idx = 0;
		int boundary_type_label_data_idx = 0;
		if (use_gt_ && (use_hn_ || use_hp_) && !rnd_fp_patch_ && !rnd_mosaic_) {
			for (int n = 0; n < batchsz_; n++) {
				std::string db_name = data[n]->db_name_;
				std::string file_name = data[n]->file_name_;
				std::string key = db_name + "/" + file_name;
				if (use_hn_ && (Caffe::hard_negative_pool().find(key) != Caffe::hard_negative_pool().end())) {
					int num_hn = Caffe::hard_negative_pool()[key].size() / 4;
					for (int i = 0; i < num_hn; i++) {
						db::ObjectGT hn_gt;
						db::Box2D hn_box;
						hn_box.x1 = Caffe::hard_negative_pool()[key][i * 4 + 0];
						hn_box.y1 = Caffe::hard_negative_pool()[key][i * 4 + 1];
						hn_box.x2 = Caffe::hard_negative_pool()[key][i * 4 + 2];
						hn_box.y2 = Caffe::hard_negative_pool()[key][i * 4 + 3];
						//printf("hn_box[%d/%d]: %f, %f, %f, %f\n", i + 1, num_hn, hn_box.x1, hn_box.y1, hn_box.x2, hn_box.y2);
						hn_box.label = -100001;
						hn_gt.box_2d_idx_ = data[n]->box_2d_.size();
						data[n]->box_2d_.push_back(hn_box);
						data[n]->object_gt_.push_back(hn_gt);
					}
				}
				if (use_hp_ && (Caffe::hard_positive_pool().find(key) != Caffe::hard_positive_pool().end())) {
					int num_hp = Caffe::hard_positive_pool()[key].size() / 4;
					for (int i = 0; i < num_hp; i++) {
						db::ObjectGT hp_gt;
						db::Box2D hp_box;
						hp_box.x1 = Caffe::hard_positive_pool()[key][i * 4 + 0];
						hp_box.y1 = Caffe::hard_positive_pool()[key][i * 4 + 1];
						hp_box.x2 = Caffe::hard_positive_pool()[key][i * 4 + 2];
						hp_box.y2 = Caffe::hard_positive_pool()[key][i * 4 + 3];
						//printf("hp_box[%d/%d]: %f, %f, %f, %f\n", i + 1, num_hp, hp_box.x1, hp_box.y1, hp_box.x2, hp_box.y2);
						hp_box.label = -100002;
						hp_gt.box_2d_idx_ = data[n]->box_2d_.size();
						data[n]->box_2d_.push_back(hp_box);
						data[n]->object_gt_.push_back(hp_gt);
					}
				}
			}
		}
		if (use_gt_ && use_maintain_aspect_ratio_) {
			for (int n = 0; n < batchsz_; n++) {
				cv::Mat new_img;
				std::vector<db::Box2D> new_box2d;
				std::vector<db::ObjectGT> new_object_gt;
				if (maintain_aspect_ratio(data[n], new_img, new_box2d, new_object_gt, maintain_aspect_ratio_width_, maintain_aspect_ratio_height_, aspect_ratio_ioa_threshold_)) {
					data[n]->img_ = new_img;
					data[n]->box_2d_ = new_box2d;
					data[n]->object_gt_ = new_object_gt;
				}
			}
		}

		if (use_gt_ && rnd_fp_patch_) {
			for (int n = 0; n < batchsz_; n++) {
				cv::Mat gt_mask_map = cv::Mat::zeros(data[n]->img_.rows, data[n]->img_.cols, CV_8UC1);
				for (int i = 0; i < data[n]->box_2d_.size(); i++) {
					int gt_width = data[n]->box_2d_[i].x2 - data[n]->box_2d_[i].x1 + 1;
					int gt_height = data[n]->box_2d_[i].y2 - data[n]->box_2d_[i].y1 + 1;
					cv::Mat dst_roi = gt_mask_map(cv::Rect(data[n]->box_2d_[i].x1, data[n]->box_2d_[i].y1, gt_width, gt_height));
					cv::Mat(gt_height, gt_width, CV_8UC1, 255).copyTo(dst_roi);
				}

				int n_patch = rand() % (fp_patch_max_n_ - fp_patch_min_n_ + 1) + fp_patch_min_n_;
				for (int m = 0; m < n_patch; m++) {
					const db::UDBPoint* false_positive_point = false_positive_cursor_->GetPoint();
					false_positive_cursor_->Next();
					db::UDBDatum false_positive_datum;
					get_datum(false_positive_point, &false_positive_datum);
					cv::Mat false_positive_img = false_positive_datum.img_;

					int random_x, random_y, available_width, available_height;
					if (select_space_randomly(gt_mask_map, false_positive_img.cols, false_positive_img.rows, (int)(data[n]->img_.cols * fp_patch_range_x_min_), (int)(data[n]->img_.rows * fp_patch_range_y_min_), (int)(data[n]->img_.cols * fp_patch_range_x_max_) - 1, (int)(data[n]->img_.rows * fp_patch_range_y_max_) - 1, random_x, random_y, available_width, available_height)) {
						cv::Mat dst_fp_img_;
						cv::Mat dst_roi;
						dst_roi = data[n]->img_(cv::Rect(random_x, random_y, available_width, available_height));
						resize(false_positive_img, dst_fp_img_, cv::Size(available_width, available_height), 0, 0, cv::INTER_CUBIC);
						dst_fp_img_.copyTo(dst_roi);

						db::ObjectGT fp_gt;
						db::Box2D fp_box;
						fp_box.x1 = random_x;
						fp_box.y1 = random_y;
						fp_box.x2 = random_x + available_width - 1;
						fp_box.y2 = random_y + available_height - 1;
						fp_box.label = -100001;
						fp_gt.box_2d_idx_ = data[n]->box_2d_.size();
						data[n]->box_2d_.push_back(fp_box);
						data[n]->object_gt_.push_back(fp_gt);

						cv::Mat mask_roi = gt_mask_map(cv::Rect(fp_box.x1, fp_box.y1, available_width, available_height));
						cv::Mat(available_height, available_width, CV_8UC1, 128).copyTo(mask_roi);
					}
				}

				/* display cutpaste
				cv::Mat img_with_objs = data[n]->img_.clone();
				for (int i = 0; i < data[n]->box_2d_.size(); i++) {
				  cv::Scalar color;
				  db::Box2D box2d_ = data[n]->box_2d_[i];
				  int label = box2d_.label;
				  switch (label) {
					case 1: case -1: color = cv::Scalar(0, 255, 0); break;
					case 2: case -2: color = cv::Scalar(0, 255, 255); break;
					case 3: case -3: color = cv::Scalar(0, 0, 255); break;
					case 4: case -4: color = cv::Scalar(0, 93, 187); break;
					case 5: case -5: color = cv::Scalar(133, 21, 199); break;
					case 6: case -6: color = cv::Scalar(255, 180, 105); break;
					case -100000: color = cv::Scalar(255, 255, 255); break;
					case -100001: color = cv::Scalar(255, 0, 0); break;
				  }
				  int thick = label > 0 ? 2 : 1;
				  int x = box2d_.x1;
				  int y = box2d_.y1;
				  int w = round(box2d_.x2 - box2d_.x1 + 1);
				  int h = round(box2d_.y2 - box2d_.y1 + 1);
				  cv::rectangle(img_with_objs, cv::Rect(x, y, w, h), color, thick);
				}
				cv::imshow("cutpaste", img_with_objs);
				cv::waitKey(1);
				*/
			}
		}

		Dtype mean_noises[3];
		switch (mean_noise_mode_) {
		case UDBDataParameter_MeanNoiseMode_UNIFORM: // U(-std, +std)
			caffe_rng_uniform(3, Dtype(-1), Dtype(1), mean_noises);
			mean0_noise_ = (mean0_noise_std_ > Dtype(0)) ? mean0_noise_std_ * mean_noises[0] : Dtype(0);
			mean1_noise_ = (mean1_noise_std_ > Dtype(0)) ? mean1_noise_std_ * mean_noises[1] : Dtype(0);
			mean2_noise_ = (mean2_noise_std_ > Dtype(0)) ? mean2_noise_std_ * mean_noises[2] : Dtype(0);
			break;
		case UDBDataParameter_MeanNoiseMode_NORMAL: // N(0, std)
			caffe_rng_gaussian(3, Dtype(0), Dtype(1), mean_noises);
			mean0_noise_ = (mean0_noise_std_ > Dtype(0)) ? mean0_noise_std_ * mean_noises[0] : Dtype(0);
			mean1_noise_ = (mean1_noise_std_ > Dtype(0)) ? mean1_noise_std_ * mean_noises[1] : Dtype(0);
			mean2_noise_ = (mean2_noise_std_ > Dtype(0)) ? mean2_noise_std_ * mean_noises[2] : Dtype(0);
			break;
		default:
			mean0_noise_ = Dtype(0);
			mean1_noise_ = Dtype(0);
			mean2_noise_ = Dtype(0);
			break;
		}

		for (int n = 0; n < data.size(); n++) {
			const db::UDBDatum& cur_data = *data[n];

			cv::Mat img = cur_data.img_;
			const DistortionParameter& distort_param = this->layer_param_.udb_data_param().distort_param();
			const NoiseParameter& noise_param = this->layer_param_.udb_data_param().noise_param();
			bool is_distort = distort_param.brightness_prob() || distort_param.contrast_prob() || distort_param.hue_prob() || distort_param.saturation_prob() || distort_param.random_order_prob();
			bool is_noise = noise_param.prob();
			if ((is_distort || is_noise) && caffe_rng_rand() % 4 == 0) {
				if (is_distort && caffe_rng_rand() % 2 == 0)
					img = ApplyDistort(img, distort_param);
				if (is_noise && caffe_rng_rand() % 2 == 0)
					img = ApplyNoise(img, noise_param);
			}
		}

		if (use_gt_ && rnd_mosaic_) {
			for (int n = 0; n < batchsz_; n++) {
				if (data.size() >= batchsz_ + (n + 1) * 3) {
					//cv::Mat new_img = data[n]->img_.clone();
					cv::Mat new_img = cv::Mat::zeros(data[n]->img_.rows, data[n]->img_.cols, CV_8UC3);
					std::vector<db::Box2D> new_box2d;
					std::vector<db::ObjectGT> new_object_gt;

					int mosaic_idx1 = batchsz_ + n;
					int mosaic_idx2 = batchsz_ + n * 2;
					int mosaic_idx3 = batchsz_ + n * 3;

					int min_padding_x = new_img.cols * mosaic_padding_min_;
					int min_padding_y = new_img.rows * mosaic_padding_min_;
					int pivot_x = rand() % (new_img.cols - 2 * min_padding_x) + min_padding_x;
					int pivot_y = rand() % (new_img.rows - 2 * min_padding_y) + min_padding_y;

					crop_image_randomly(0, 0, pivot_x - 1 - mosaic_boundary_, pivot_y - 1 - mosaic_boundary_, data[n], new_img, new_box2d, new_object_gt);
					crop_image_randomly(pivot_x + mosaic_boundary_, 0, new_img.cols - 1, pivot_y - 1 - mosaic_boundary_, data[mosaic_idx1], new_img, new_box2d, new_object_gt);
					crop_image_randomly(0, pivot_y + mosaic_boundary_, pivot_x - 1 - mosaic_boundary_, new_img.rows - 1, data[mosaic_idx2], new_img, new_box2d, new_object_gt);
					crop_image_randomly(pivot_x + mosaic_boundary_, pivot_y + mosaic_boundary_, new_img.cols - 1, data[n]->img_.rows - 1, data[mosaic_idx3], new_img, new_box2d, new_object_gt);

					if (new_object_gt.size() > 0) {
						data[n]->img_ = new_img;
						data[n]->box_2d_ = new_box2d;
						data[n]->object_gt_ = new_object_gt;
					}

					/* display mosaic
					cv::Mat img_with_objs = data[n]->img_.clone();
					for (int i = 0; i < data[n]->box_2d_.size(); i++) {
					  cv::Scalar color;
					  db::Box2D box2d_ = data[n]->box_2d_[i];
					  int label = box2d_.label;
					  switch (label) {
						case 1: case -1: color = cv::Scalar(0, 255, 0); break;
						case 2: case -2: color = cv::Scalar(0, 255, 255); break;
						case 3: case -3: color = cv::Scalar(0, 0, 255); break;
						case 4: case -4: color = cv::Scalar(0, 93, 187); break;
						case 5: case -5: color = cv::Scalar(133, 21, 199); break;
						case 6: case -6: color = cv::Scalar(255, 180, 105); break;
						case -100000: color = cv::Scalar(255, 255, 255); break;
						case -100001: color = cv::Scalar(255, 0, 0); break;
					  }
					  int thick = label > 0 ? 2 : 1;
					  int x = box2d_.x1;
					  int y = box2d_.y1;
					  int w = round(box2d_.x2 - box2d_.x1 + 1);
					  int h = round(box2d_.y2 - box2d_.y1 + 1);
					  cv::rectangle(img_with_objs, cv::Rect(x, y, w, h), color, thick);
					}
					cv::imshow("mosaic", img_with_objs);
					cv::waitKey(0);
					*/
				}
			}
		}

		if (patch.size() > 0) {
			for (int n = 0; n < batchsz_; n++) {
				const int w = data[n]->img_.cols;
				const int h = data[n]->img_.rows;
				int start_i = 0;
				int start_x, start_y, end_x, end_y;
				std::vector<int> w_info(w * h, -2);
				std::vector<int> h_info(w * h, -2);
				for (int l = 0; l < patch.size(); l++) {
					if (!patch[l]->is_open())
						continue;
					for (int p = 0; p < patch_aug_num[l]; p++) {
						if (start_i < data[n]->box_2d_.size()) {
							for (int i = start_i; i < data[n]->box_2d_.size(); i++) {
								if (data[n]->box_2d_[i].label != -100000 && data[n]->box_2d_[i].label != -100001 && data[n]->box_2d_[i].label != -100002) {
									for (int y = data[n]->box_2d_[i].y1; y <= data[n]->box_2d_[i].y2; y++) {
										for (int x = data[n]->box_2d_[i].x1; x <= data[n]->box_2d_[i].x2; x++) {
											w_info[y * w + x] = -1;
											h_info[y * w + x] = -1;
										}
									}
								}
							}
							if (start_i == 0) {
								for (int y = 0; y < h; y++) {
									for (int x = 0; x < w; x++) {
										if (w_info[y * w + x] == -2) {
											w_info[y * w + x] = 0;
											int x0 = x;
											for (; x0 < w; x0++) {
												if (w_info[y * w + x0] == -1)
													break;
											}
											for (int x1 = x; x1 < x0; x1++) {
												w_info[y * w + x1] = x0 - x1;
											}
										}
										if (h_info[y * w + x] == -2) {
											h_info[y * w + x] = 0;
											int y0 = y;
											for (; y0 < h; y0++) {
												if (h_info[y0 * w + x] == -1)
													break;
											}
											for (int y1 = y; y1 < y0; y1++) {
												h_info[y1 * w + x] = y0 - y1;
											}
										}
									}
								}
							}
							else {
								for (int y = start_y; y >= 0; y--) {
									for (int x = start_x; x < end_x; x++) {
										if (h_info[y * w + x] == -1)
											break;
										h_info[y * w + x] = y - start_y + 1;
									}
								}
								for (int x = start_x; x >= 0; x--) {
									for (int y = start_y; y < end_y; y++) {
										if (w_info[y * w + x] == -1)
											break;
										w_info[y * w + x] = w - start_x + 1;
									}
								}
							}
							start_i = data[n]->box_2d_.size();
						}
						const int pidx = caffe_rng_rand() % patch_pointer[l].size();
						patch[l]->seekg(patch_pointer[l][pidx], patch[l]->beg);
						int pw = 0, ph = 0, x1 = 0, y1 = 0, x2 = 0, y2 = 0;
						patch[l]->read((char*)&pw, sizeof(int));
						patch[l]->read((char*)&ph, sizeof(int));
						patch[l]->read((char*)&x1, sizeof(int));
						patch[l]->read((char*)&y1, sizeof(int));
						patch[l]->read((char*)&x2, sizeof(int));
						patch[l]->read((char*)&y2, sizeof(int));
						if (pw >= w || ph >= h) {
							continue;
						}
						cv::Mat pimg(ph, pw, CV_8UC3);
						patch[l]->read((char*)pimg.data, pw * ph * 3 * sizeof(unsigned char));
						//cv::imshow("test", pimg);
						//cv::waitKey(0);
						int rnd_x, rnd_y, t, retry;
						for (t = 0; t < 100; t++) {
							rnd_x = caffe_rng_rand() % (w - pw);
							rnd_y = caffe_rng_rand() % (h - ph);
							retry = false;
							for (int rx = rnd_x; rx < rnd_x + pw; rx++) {
								if (ph > h_info[rnd_y * w + rx]) {
									retry = true;
									break;
								}
							}
							if (retry)
								continue;
							for (int ry = rnd_y; ry < rnd_y + ph; ry++) {
								if (pw > w_info[ry * w + rnd_x]) {
									retry = true;
									break;
								}
							}
							if (retry)
								continue;
							break;
						}
						if (retry)
							continue;
						db::ObjectGT aug_gt;
						db::Box2D aug_box;
						aug_box.x1 = rnd_x + x1;
						aug_box.y1 = rnd_y + y1;
						aug_box.x2 = rnd_x + x2;
						aug_box.y2 = rnd_y + y2;
						aug_box.label = patch_class_index[l];
						aug_gt.box_2d_idx_ = data[n]->box_2d_.size();
						start_x = aug_box.x1;
						start_y = aug_box.y1;
						end_x = aug_box.x2 + 1;
						end_y = aug_box.y2 + 1;
						data[n]->box_2d_.push_back(aug_box);
						data[n]->object_gt_.push_back(aug_gt);
						cv::Mat proi = data[n]->img_(cv::Rect(rnd_x, rnd_y, pw, ph));
						for (int r = 0; r < ph; r++) {
							cv::Vec3b* pimg_data = pimg.ptr<cv::Vec3b>(r);
							cv::Vec3b* proi_data = proi.ptr<cv::Vec3b>(r);
							for (int c = 0; c < pw; c++) {
								const float alpha = (std::max(std::max(std::abs(float(c) / pw - 0.5f), std::abs(float(r) / ph - 0.5f)), 0.2f) - 0.2f) / 0.3f;
								const float beta = 1 - alpha;
								proi_data[c][0] = proi_data[c][0] * alpha + pimg_data[c][0] * beta;
								proi_data[c][1] = proi_data[c][1] * alpha + pimg_data[c][1] * beta;
								proi_data[c][2] = proi_data[c][2] * alpha + pimg_data[c][2] * beta;
							}
						}
					}
					//cv::imshow("test", data[n]->img_);
					//cv::waitKey(0);
				}
			}
		}

		for (int n = 0; n < batchsz_; n++) {
			for (int i = 0; i < data[n]->box_2d_.size(); i++) {
				if (data[n]->box_2d_[i].label == -100000) {
					num_all_dcs++;
					if (use_ignore_as_rpn_gt_)
						num_all_gts++;
				}
				else if (data[n]->box_2d_[i].label == -100001) {
					num_all_hns++;
				}
				else if (data[n]->box_2d_[i].label == -100002) {
					num_all_hps++;
				}
				else {
					num_all_gts++;
					if (use_gt_ && use_gt_quad_)
						num_all_gts_quad++;
					if (data[n]->box_2d_[i].label < 0) {
						num_all_dcs++;
					}
				}
			}
			num_all_gts_3d += data[n]->box_3d_.size();
			num_all_gts_new_3d += data[n]->box_new_3d_.size();
			for (int i = 0; i < data[n]->object_gt_.size(); i++) {
				if ((post_type_ == POST_V1 && (data[n]->object_gt_[i].box_2d_idx_ != -1 || data[n]->object_gt_[i].box_3d_idx_ != -1)) ||
					(post_type_ == POST_V2 && (data[n]->object_gt_[i].box_2d_idx_ != -1 || data[n]->object_gt_[i].box_new_3d_idx_ != -1)) ||
					(post_type_ >= POST_V3 && (data[n]->object_gt_[i].box_2d_idx_ != -1 || data[n]->object_gt_[i].box_new_3d_idx_ != -1))) {
					num_all_gts_ex++;
				}
			}
		}

		if (use_blank_roi_ &&
			((use_gt_ && num_all_gts == 0) ||
			(use_gt_quad_ && num_all_gts_quad == 0) ||
				(use_gt_3d_ && num_all_gts_3d == 0) ||
				(use_gt_new_3d_ && num_all_gts_new_3d == 0) ||
				(use_gt_ex_ && num_all_gts_ex == 0)))
			return false;

		CHECK(!use_img_ || present_img) << "You should provide images.";
		CHECK(!use_gt_ || num_all_gts) << "You should provide gt rois.";
		CHECK(!use_gt_quad_ || num_all_gts_quad) << "You should provide gt_quad rois.";
		CHECK(!use_seg_ || present_seg) << "You should provide segmentation labels.";
		CHECK(!use_scene_lbl_ || present_scene_lbl) << "You should provide scene labels.";
		CHECK(!use_failsafe_ || present_failsafe) << "You should provide failsafe labels.";
		CHECK(!use_gt_3d_ || num_all_gts_3d) << "You should provide 3d gt rois.";
		CHECK(!use_gt_new_3d_ || num_all_gts_new_3d) << "You should provide new 3d gt rois.";
		CHECK(!use_gt_ex_ || num_all_gts_ex) << "You should provide attribute gt rois.";
		CHECK(!use_gt_ex_ || post_type_ >= POST_V1) << "gt_roi_ex should be with post_type >= 2.";
		CHECK(fixed_aspect_ == 0 || (!num_all_gts && !num_all_gts_3d && !num_all_gts_new_3d && !num_all_gts_ex)) << "fixed_aspect is not allowed with gt_roi or gt_roi_3d or gt_roi_ex";
		CHECK(image_roi_.size() == 0 || (!num_all_gts && !num_all_gts_3d && !num_all_gts_new_3d && !num_all_gts_ex)) << "image_roi is not allowed with gt_roi or gt_roi_3d or gt_roi_ex";
		CHECK(image_roi_.size() == 0 || use_fixed_size_) << "image_roi is only allowed with use_fixed_size";
		CHECK(use_gt_ || !use_closeness_) << "closeness must be used with GT_ROIS";
		CHECK(use_ego_out_ || !use_rnd_perspective_) << "perspective transform must be defined with ego_out";
		CHECK(!use_ego_out_ || (use_ego_out_ && !rnd_affine_ && !rnd_crop_)) << "ego must not be used with rnd_affine or rnd_crop, but both of them are implemented with fixed_perspective.";
		CHECK(!rnd_mosaic_ || !rnd_affine_) << "rnd_mosaic must not be used with rnd_affine";
		CHECK(!use_meta_info_ || present_meta_info) << "You should provide meta info.";

		int resized_width, resized_height;
		if (use_fixed_size_) {
			int fixed_scale_index = -1;
			if (!round_robin_fixed_scale_) {
				if (udb_index_ == -1) {
					//printf("lead0: %d\n", udb_index_);
					fixed_scale_index = caffe_rng_rand() % fixed_scale_.size();
					for (int i = 0; i < udb_layer_count_; i++) {
						while (fixed_scale_index_[i].size() != 0) {
							boost::chrono::milliseconds(1);
						}
					}
					for (int i = 0; i < udb_layer_count_; i++) {
						fixed_scale_index_[i].push(fixed_scale_index);
					}
					//printf("lead1: %d\n", fixed_scale_index);
				}
				else {
					//printf("follow0: %d\n", udb_index_);
					fixed_scale_index = fixed_scale_index_[udb_index_].pop();
					//printf("follow1: %d\n", fixed_scale_index);
				}
			}
			else {
				fixed_scale_index = round_robin_fixed_scale_index_;
				round_robin_fixed_scale_index_ = (round_robin_fixed_scale_index_ + 1) % fixed_scale_.size();
			}
			resized_width = fixed_scale_[fixed_scale_index][0] / multiple_ * multiple_;
			resized_height = fixed_scale_[fixed_scale_index][1] / multiple_ * multiple_;
			batch->data_.Reshape(batchsz_, input_channel_[input_type_], resized_height, resized_width);
			for (int i = 0; i < num_segments_; i++) {
				if (!segment_label_resize_[i]) {
					batch->seg_[i]->Reshape(batchsz_, 1 * (segment_map_scale_[i] * segment_map_scale_[i]), resized_height / segment_map_scale_[i], resized_width / segment_map_scale_[i]);
				}
				else {
					batch->seg_[i]->Reshape(batchsz_, 1, resized_height / segment_map_scale_[i], resized_width / segment_map_scale_[i]);
				}
			}
			if (use_edge_) {
				batch->edge_.Reshape(batchsz_, 1, resized_height / segment_map_scale_[0], resized_width / segment_map_scale_[0]);
			}
			if (use_tlr_blobReg_)
				batch->tlr_blobReg_.Reshape(batchsz_, 6, resized_height, resized_width);
		}

		if (use_gt_)
			batch->gt_rois_.Reshape(num_all_gts, 6);
		if (use_gt_quad_)
			batch->gt_rois_quad_.Reshape(num_all_gts_quad, 10);
		if (use_gt_3d_)
			batch->gt_rois_3d_.Reshape(num_all_gts_3d, 10);
		if (use_gt_new_3d_)
			batch->gt_rois_new_3d_.Reshape(num_all_gts_new_3d, 17);
		if (use_gt_ex_)
			batch->gt_rois_ex_.Reshape(num_all_gts_ex, 32);
		if (use_scene_lbl_)
			batch->label_.Reshape(batchsz_, 3);
		if (use_meta_info_)
			batch->meta_.Reshape(batchsz_, 6);
		if (use_failsafe_)
			batch->failsafe_.Reshape(batchsz_, 1);
		if (use_img_rois_)
			batch->img_rois_.Reshape(batchsz_, 5);
		if (use_info_)
			batch->info_.Reshape(batchsz_, 12);
		if (use_dc_) {
			if (num_all_dcs) {
				batch->dc_rois_.Reshape(num_all_dcs, 5);
			}
			else {
				num_all_dcs = 1;
				batch->dc_rois_.Reshape(num_all_dcs, 5);
				caffe_set(5, (Dtype)0, batch->dc_rois_.mutable_cpu_data());
			}
		}
		if (use_hn_) {
			if (num_all_hns) {
				batch->hn_rois_.Reshape(num_all_hns, 5);
			}
			else {
				num_all_hns = 1;
				batch->hn_rois_.Reshape(num_all_hns, 5);
				caffe_set(5, (Dtype)0, batch->hn_rois_.mutable_cpu_data());
			}
		}
		if (use_hp_) {
			if (num_all_hps) {
				batch->hp_rois_.Reshape(num_all_hps, 5);
			}
			else {
				num_all_hps = 1;
				batch->hp_rois_.Reshape(num_all_hps, 5);
				caffe_set(5, (Dtype)0, batch->hp_rois_.mutable_cpu_data());
			}
		}
		if (use_gt_info_)
			batch->gt_info_.Reshape(batchsz_, 3, 512);
		if (use_ego_out_)
			batch->ego_out_.Reshape(batchsz_, (int)ego_grid_, 2);
		if (use_closeness_)
			batch->closeness_.Reshape(batchsz_, 1);

		if (use_tsr_cls_)
			batch->tsr_cls_.Reshape(batchsz_, 1);

		if (use_tlr_cls_)
			batch->tlr_cls_.Reshape(batchsz_, 1);
		if (use_tlr_blobs_)
			batch->tlr_blobs_.Reshape(batchsz_, 1);

		if (use_lane_type_label_) {
			int num_total_lane_type_label = 0;
			for (int n = 0; n < batchsz_; n++) {
				num_total_lane_type_label += data[n]->lane_type_label_.size();
			}
			batch->lane_type_label_.Reshape(1 + num_total_lane_type_label);
			batch->lane_type_label_.mutable_cpu_data()[0] = 5; // lane_class_num
		}
		if (use_boundary_type_label_) {
			int num_total_boundary_type_label = 0;
			for (int n = 0; n < batchsz_; n++) {
				num_total_boundary_type_label += data[n]->boundary_type_label_.size();
			}
			batch->boundary_type_label_.Reshape(1 + num_total_boundary_type_label);
			batch->boundary_type_label_.mutable_cpu_data()[0] = 2; // boundary_class_num
		}

		for (int n = 0; n < batchsz_; n++) {
			const db::UDBDatum& cur_data = *data[n];

			cv::Mat img = cur_data.img_;
			int width = img.cols;
			int height = img.rows;

			bool mirror = rnd_mirror_ ? caffe_rng_rand() % 2 : false;
			bool color_aug = color_aug_ ? caffe_rng_rand() % 2 : false;



			int pers_idx = use_rnd_perspective_ ? caffe_rng_rand() % fixed_perspective_.size() + 1 : 0;
			int shadow_idx = use_shadow_ ? caffe_rng_rand() % shadow_anchor_.size() + 1 : 0;

			Dtype* lane_type_label_data = batch->lane_type_label_.count() ? batch->lane_type_label_.mutable_cpu_data() + 1 + lane_type_label_data_idx : NULL;
			if (lane_type_label_data) {
				int cur_data_size = cur_data.lane_type_label_.size();
				for (int nn = 0; nn < cur_data_size; nn++) {
					lane_type_label_data[nn] = cur_data.lane_type_label_[nn];
				}
				lane_type_label_data_idx += cur_data_size;
			}
			Dtype* boundary_type_label_data = batch->boundary_type_label_.count() ? batch->boundary_type_label_.mutable_cpu_data() + 1 + boundary_type_label_data_idx : NULL;
			if (boundary_type_label_data) {
				int cur_data_size = cur_data.boundary_type_label_.size();
				for (int nn = 0; nn < cur_data_size; nn++) {
					boundary_type_label_data[nn] = cur_data.boundary_type_label_[nn];
				}
				boundary_type_label_data_idx += cur_data_size;
			}

			// parameters for perspective transform
			cv::Mat H;
			cv::Mat H2;
			float org_width = img.cols;
			float org_height = img.rows;
			std::vector<cv::Point2f> pers_org_pts;
			std::vector<cv::Point2f> pers_dst_pts;

			if (pers_idx) {
				// perspective transform applied to following 4 points
				// top left point
				pers_org_pts.push_back(cv::Point2f(0, 0));
				// bottom left point
				pers_org_pts.push_back(cv::Point2f(0, org_height));
				// top right
				pers_org_pts.push_back(cv::Point2f(org_width, 0));
				// bottom right
				pers_org_pts.push_back(cv::Point2f(org_width, org_height));

				//pesrpective augmentation parameters
				float pscale = 1;
				float pcropx = 0;
				float pcropy = 0;

				// we use vanishing points as center point in perspective augmentation
				float cx = cur_data.vp_x_;
				float cy = cur_data.vp_y_;

				// use random scaling
				if (use_rnd_pers_scale_) {
					pscale = min<float>(1, pers_scale_min_ + ((Dtype)caffe_rng_rand() / UINT_MAX));
				}
				else {
					pscale = 1;
				}
				// use random center movements
				if (use_rnd_pers_center_) {
					cx = cx + (((Dtype)caffe_rng_rand() / UINT_MAX) - 0.5) * pers_center_max_;
					cy = cy + (((Dtype)caffe_rng_rand() / UINT_MAX) - 0.5) * pers_center_max_;
				}

				// get fixed_perspective values
				float tdx = fixed_perspective_[pers_idx - 1][0] * pscale;
				float tdy = fixed_perspective_[pers_idx - 1][1] * pscale;
				float bdx = fixed_perspective_[pers_idx - 1][2] * pscale;
				float bdy = fixed_perspective_[pers_idx - 1][3] * pscale;

				// destination positions for 4 corner points

				// bottom left x
				float bxl = (cx - cx * bdx) * org_width;
				// bottom right x
				float bxr = (cx + (1 - cx) * bdx) * org_width;

				// top left x
				float txl = (cx - cx * tdx) * org_width;
				// top right x
				float txr = (cx + (1 - cx) * tdx) * org_width;

				// top y
				float ty = (cy - cy * tdy) * org_height;
				// bottom y
				float by = (cy + (1 - cy) * bdy) * org_height;;

				// top left point
				pers_dst_pts.push_back(cv::Point2f(txl, ty));
				// bottom left point
				pers_dst_pts.push_back(cv::Point2f(bxl, by));
				// top right point
				pers_dst_pts.push_back(cv::Point2f(txr, ty));
				// bottom right point
				pers_dst_pts.push_back(cv::Point2f(bxr, by));


				H = cv::findHomography(pers_dst_pts, pers_org_pts, 0);
			}


			Dtype* ego_out = NULL;

			if (use_ego_out_) {
				vector<cv::Point2f> org_pts;
				vector<cv::Point2f> dst_pts;
				if (pers_idx) {
					// transform all ego lane points by perspective transform
					org_pts.insert(org_pts.end(), cur_data.ego_xy_L_.begin(), cur_data.ego_xy_L_.end());
					org_pts.insert(org_pts.end(), cur_data.ego_xy_R_.begin(), cur_data.ego_xy_R_.end());
					for (int i = 0; i < org_pts.size(); i++) {
						org_pts[i].x *= org_width;
						org_pts[i].y *= org_height;
					}
					dst_pts.resize(org_pts.size());
					cv::perspectiveTransform(org_pts, dst_pts, H);
				}
				else {
					dst_pts = cur_data.ego_xy_L_;
					dst_pts.insert(dst_pts.end(), cur_data.ego_xy_R_.begin(), cur_data.ego_xy_R_.end());
					for (int i = 0; i < dst_pts.size(); i++) {
						dst_pts[i].x *= org_width;
						dst_pts[i].y *= org_height;
					}
				}
				//dst_pts are the points to be ego grid gt

				vector<vector<int>> cnt_gridL(ego_grid_);
				vector<vector<int>> cnt_gridR(ego_grid_);


				ego_out = batch->ego_out_.mutable_cpu_data() + n * (batch->ego_out_.shape(1) * batch->ego_out_.shape(2));
				//cur_data.ego_xy_ = height * 2

				int gridx = 0;
				int gridy = 0;

				// count ego lane pixels for each grid of each lane(left,right)
				for (int i = 0; i < dst_pts.size(); i++) {

					if (dst_pts[i].x >= 0 && dst_pts[i].x < org_width && dst_pts[i].y >= 0 && dst_pts[i].y < org_height) {
						gridy = (int)(dst_pts[i].y / org_height * (float)ego_grid_);
						gridx = (int)(dst_pts[i].x / org_width * (float)(ego_grid_ - 1));


						if (i < cur_data.ego_xy_L_.size()) {
							if (cnt_gridL[gridy].size() == 0) {
								cnt_gridL[gridy].resize(ego_grid_ - 1, 0);
							}
							cnt_gridL[gridy][gridx]++;
						}
						else {
							if (cnt_gridR[gridy].size() == 0) {
								cnt_gridR[gridy].resize(ego_grid_ - 1, 0);
							}
							cnt_gridR[gridy][gridx]++;
						}
					}


				}

				// find the ego grid which has maximum pixel count for each row
				for (int h = 0; h < ego_grid_; h++) {

					int maxval[2];
					int maxidx[2];
					maxidx[0] = -1;
					maxidx[1] = -1;
					maxval[0] = -1;
					maxval[1] = -1;


					if (cnt_gridL[h].size() > 0) {
						for (int i = 0; i < (int)ego_grid_ - 1; i++) {

							if (maxval[0] < cnt_gridL[h][i]) {
								maxval[0] = cnt_gridL[h][i];
								maxidx[0] = i;
							}
						}
					}
					if (cnt_gridR[h].size() > 0) {
						for (int i = 0; i < (int)ego_grid_ - 1; i++) {

							if (maxval[1] < cnt_gridR[h][i]) {
								maxval[1] = cnt_gridR[h][i];
								maxidx[1] = i;
							}
						}
					}



					if (!mirror) {
						if (maxval[0] == -1) {
							ego_out[h * 2] = 0;
						}
						else {
							ego_out[h * 2] = maxidx[0] + 1;
						}
						if (maxval[1] == -1) {
							ego_out[h * 2 + 1] = 0;
						}
						else {
							ego_out[h * 2 + 1] = maxidx[1] + 1;
						}
					}
					else {
						if (maxval[1] == -1) {
							ego_out[h * 2] = 0;
						}
						else {
							ego_out[h * 2] = ego_grid_ - maxidx[1] - 1;
						}
						if (maxval[0] == -1) {
							ego_out[h * 2 + 1] = 0;
						}
						else {
							ego_out[h * 2 + 1] = ego_grid_ - maxidx[0] - 1;
						}
					}

				}
			}



			Dtype affine_param[8] = { 1, 0, 0, 1, 1, 0, 0, 1 };

			Dtype* gt_rois = NULL;
			if (use_gt_) {
				gt_rois = batch->gt_rois_.mutable_cpu_data() + batch->gt_rois_.offset(gt_roi_index);
			}
			Dtype* gt_rois_quad = NULL;
			if (use_gt_quad_) {
				gt_rois_quad = batch->gt_rois_quad_.mutable_cpu_data() + batch->gt_rois_quad_.offset(gt_roi_quad_index);
			}
			Dtype* gt_rois_3d = NULL;
			if (use_gt_3d_) {
				gt_rois_3d = batch->gt_rois_3d_.mutable_cpu_data() + batch->gt_rois_3d_.offset(gt_roi_3d_index);
			}
			Dtype* gt_rois_new_3d = NULL;
			if (use_gt_new_3d_) {
				gt_rois_new_3d = batch->gt_rois_new_3d_.mutable_cpu_data() + batch->gt_rois_new_3d_.offset(gt_roi_new_3d_index);
			}
			Dtype* gt_rois_ex = NULL;
			if (use_gt_ex_) {
				gt_rois_ex = batch->gt_rois_ex_.mutable_cpu_data() + batch->gt_rois_ex_.offset(gt_roi_ex_index);
			}
			Dtype* dc_rois = NULL;
			if (use_dc_) {
				dc_rois = batch->dc_rois_.mutable_cpu_data() + batch->dc_rois_.offset(dc_roi_index);
			}
			Dtype* hn_rois = NULL;
			if (use_hn_) {
				hn_rois = batch->hn_rois_.mutable_cpu_data() + batch->hn_rois_.offset(hn_roi_index);
			}
			Dtype* hp_rois = NULL;
			if (use_hp_) {
				hp_rois = batch->hp_rois_.mutable_cpu_data() + batch->hp_rois_.offset(hp_roi_index);
			}
			if (!use_fixed_size_) {
				resized_width = 0;
				resized_height = 0;
			}
			int crop_x, crop_y, crop_w, crop_h;
			Dtype im_scale_x, im_scale_y;
			int num_gts, num_gts_quad, num_gts_3d, num_gts_new_3d, num_gts_ex, num_dcs, num_hns, num_hps;


			Dtype* closeness = NULL;

			if (use_closeness_) {
				closeness = batch->closeness_.mutable_cpu_data() + n * batch->closeness_.shape(1);

			}
			TransformWithGT(cur_data.object_gt_, cur_data.box_2d_, cur_data.box_3d_, cur_data.box_new_3d_, cur_data.box_attribute_, cur_data.quad_2d_,
				gt_rois, gt_rois_3d, gt_rois_new_3d, gt_rois_ex, dc_rois, hn_rois, hp_rois, gt_rois_quad,
				n, mirror, width, height,
				resized_width, resized_height, crop_x, crop_y, crop_w, crop_h, im_scale_x, im_scale_y, affine_param,
				num_gts, num_gts_3d, num_gts_new_3d, num_gts_ex, num_dcs, num_hns, num_hps, num_gts_quad, closeness);



			gt_roi_index += num_gts;
			gt_roi_quad_index += num_gts_quad;
			gt_roi_3d_index += num_gts_3d;
			gt_roi_new_3d_index += num_gts_new_3d;
			gt_roi_ex_index += num_gts_ex;
			dc_roi_index += num_dcs;
			hn_roi_index += num_hns;
			hp_roi_index += num_hps;

			Dtype* _data = NULL;
			vector<Dtype*> _seg(num_segments_, NULL);
			Dtype* _edge = NULL;
			Dtype* _tlr_blobReg = NULL;

			if (use_fixed_size_) {
				_data = batch->data_.mutable_cpu_data() + batch->data_.offset(n);
				for (int i = 0; i < num_segments_; i++) {
					_seg[i] = batch->seg_[i]->mutable_cpu_data() + batch->seg_[i]->offset(n);
				}
				if (use_edge_) {
					_edge = batch->edge_.mutable_cpu_data() + batch->edge_.offset(n);
				}
				if (use_tlr_blobReg_) {
					_tlr_blobReg = batch->tlr_blobReg_.mutable_cpu_data() + batch->tlr_blobReg_.offset(n);
				}
			}
			else {
				vdata_[n]->Reshape(1, batch->data_.shape(1), resized_height, resized_width);
				_data = vdata_[n]->mutable_cpu_data();
				for (int i = 0; i < num_segments_; i++) {
					if (!segment_label_resize_[i]) {
						vseg_[n * num_segments_ + i]->Reshape(1, 1 * (segment_map_scale_[i] * segment_map_scale_[i]), resized_height / segment_map_scale_[i], resized_width / segment_map_scale_[i]);
					}
					else {
						vseg_[n * num_segments_ + i]->Reshape(1, 1, resized_height / segment_map_scale_[i], resized_width / segment_map_scale_[i]);
					}
					_seg[i] = vseg_[n * num_segments_ + i]->mutable_cpu_data();
				}
				if (use_edge_) {
					vedge_[n]->Reshape(1, 1, resized_height / segment_map_scale_[0], resized_width / segment_map_scale_[0]);
					_edge = vedge_[n]->mutable_cpu_data();
				}
				if (use_tlr_blobReg_) {
					_tlr_blobReg = vtlrReg_[n]->mutable_cpu_data();
				}
			}


			if (pers_idx) {
				cv::warpPerspective(img, img, H, cv::Size(img.cols, img.rows));
			}


			if (shadow_idx && use_shadow_) {
				if (shadow_anchor_[shadow_idx - 1][0] >= 0) {
					int lx = (int)((float)resized_width  * shadow_anchor_[shadow_idx - 1][0] * ((((Dtype)caffe_rng_rand() / UINT_MAX) - 0.5) * 0.2 + 1) / im_scale_x + crop_x);
					int ty = (int)((float)resized_height * shadow_anchor_[shadow_idx - 1][1] * ((((Dtype)caffe_rng_rand() / UINT_MAX) - 0.5) * 0.2 + 1) / im_scale_y + crop_y);
					int rx = (int)((float)resized_width  * shadow_anchor_[shadow_idx - 1][2] * ((((Dtype)caffe_rng_rand() / UINT_MAX) - 0.5) * 0.2 + 1) / im_scale_x + crop_x);
					int by = (int)((float)resized_height * shadow_anchor_[shadow_idx - 1][3] * ((((Dtype)caffe_rng_rand() / UINT_MAX) - 0.5) * 0.2 + 1) / im_scale_y + crop_y);
					float shadow_ratio = 0.25 + 0.75 * (((Dtype)caffe_rng_rand() / UINT_MAX));
					lx = std::max(std::min(lx, width - 1), (int)0);
					ty = std::max(std::min(ty, height - 1), (int)0);
					rx = std::max(std::min(rx, width - 1), (int)0);
					by = std::max(std::min(by, height - 1), (int)0);


					if (mirror) {
						int tmp = width - lx - 1;
						lx = width - rx - 1;
						rx = tmp;
					}
					if (lx > rx) {
						int tmp = lx;
						lx = rx;
						rx = tmp;
					}
					if (ty > by) {
						int tmp = ty;
						ty = by;
						by = ty;
					}
					cv::Mat imageROI = img(cv::Rect(lx, ty, rx - lx + 1, by - ty + 1));
					cv::Mat zero_box = cv::Mat::zeros(by - ty + 1, rx - lx + 1, CV_8UC3);

					// the shape of shadow is randomly selected from random polygon v.s. random box
					if (caffe_rng_rand() % 2) {
						imageROI.copyTo(zero_box);

						cv::Point polypts[1][100];
						int npts[] = { caffe_rng_rand() % 25 + 3 };

						for (int i = 0; i < npts[0]; i++) {
							polypts[0][i] = cv::Point((int)((Dtype)(zero_box.cols - 1) * ((Dtype)caffe_rng_rand() / UINT_MAX)), (int)((Dtype)(zero_box.rows - 1) * ((Dtype)caffe_rng_rand() / UINT_MAX)));
						}
						const cv::Point* polyp[1] = { polypts[0] };
						cv::fillPoly(zero_box, polyp, npts, 1, (0, 0, 0));
					}
					cv::addWeighted(imageROI, 1 - shadow_ratio, zero_box, shadow_ratio, 0., imageROI);
				}
			}

			SetImgData(img.data, _data, mirror, color_aug, width, height, resized_width, resized_height, crop_x, crop_y, im_scale_x, im_scale_y, affine_param);

			if (use_seg_) {
				for (int i = 0; i < num_segments_; i++) {
					SetSegData(cur_data.seg_.data, _seg[i], i, mirror, width, height, resized_width, resized_height, crop_x, crop_y, im_scale_x, im_scale_y, affine_param);
				}
			}

			if (use_edge_) {
				MakeEdge(cur_data.seg_, _edge, mirror, width, height, resized_width, resized_height, crop_x, crop_y, im_scale_x, im_scale_y);
			}

			if (use_scene_lbl_) {
				Dtype* _label = batch->label_.mutable_cpu_data() + batch->label_.offset(n);
				_label[0] = cur_data.scene_lbl_.time;
				_label[1] = cur_data.scene_lbl_.place;
				_label[2] = cur_data.scene_lbl_.weather;
			}

			if (use_meta_info_) {
				Dtype* _meta_info = batch->meta_.mutable_cpu_data() + batch->meta_.offset(n);
				_meta_info[0] = cur_data.meta_info_.road;
				_meta_info[1] = cur_data.meta_info_.timezone_item;
				_meta_info[2] = cur_data.meta_info_.weather_item_1;
				_meta_info[3] = cur_data.meta_info_.weather_item_2;
				_meta_info[4] = cur_data.meta_info_.weather_item_3;
				_meta_info[5] = cur_data.meta_info_.weather_item_4;
			}

			if (use_failsafe_) {
				Dtype* _failsafe = batch->failsafe_.mutable_cpu_data() + batch->failsafe_.offset(n);
				_failsafe[0] = cur_data.failsafe_;

			}
			if (use_img_rois_) {
				Dtype* _img_rois = batch->img_rois_.mutable_cpu_data() + batch->img_rois_.offset(n);
				_img_rois[0] = n;
				_img_rois[1] = 0;
				_img_rois[2] = 0;
				_img_rois[3] = resized_width;
				_img_rois[4] = resized_height;
			}

			if (use_info_) {
				Dtype* _info = batch->info_.mutable_cpu_data() + batch->info_.offset(n);
				_info[0] = width;   // ¾Æ¸¶µµ img width
				_info[1] = height;
				_info[2] = crop_x;
				_info[3] = crop_y;
				_info[4] = crop_w;
				_info[5] = crop_h;
				_info[6] = resized_width;
				_info[7] = resized_height;
				_info[8] = im_scale_x;
				_info[9] = im_scale_y;
				_info[10] = 1 / im_scale_x;
				_info[11] = 1 / im_scale_y;
			}

			if (use_gt_info_) {
				char* db_name = (char*)(batch->gt_info_.mutable_cpu_data() + batch->gt_info_.offset(n, 0));
				char* file_name = (char*)(batch->gt_info_.mutable_cpu_data() + batch->gt_info_.offset(n, 1));
				memset(db_name, 0, sizeof(Dtype) * batch->gt_info_.shape(2));
				memset(file_name, 0, sizeof(Dtype) * batch->gt_info_.shape(2));
				CHECK(cur_data.db_name_.length() < sizeof(Dtype) * batch->gt_info_.shape(2));
				CHECK(cur_data.file_name_.length() < sizeof(Dtype) * batch->gt_info_.shape(2));
				strcpy(db_name, cur_data.db_name_.c_str());
				strcpy(file_name, cur_data.file_name_.c_str());
				Dtype* transform_info = batch->gt_info_.mutable_cpu_data() + batch->gt_info_.offset(n, 2);
				transform_info[0] = mirror;
				transform_info[1] = rnd_affine_;
			}

			if (use_tsr_cls_) {
				Dtype* _tsr_cls = batch->tsr_cls_.mutable_cpu_data() + batch->tsr_cls_.offset(n);
				_tsr_cls[0] = cur_data.tsr_cls_;
			}

			if (use_tlr_cls_) {
				Dtype* _tlr_cls = batch->tlr_cls_.mutable_cpu_data() + batch->tlr_cls_.offset(n);
				_tlr_cls[0] = cur_data.tlr_cls_;
				/*caffe_set(num_objects_ * num_classes_, Dtype(0), _tlr_cls);
				for (int tlr_idx = 0; tlr_idx < num_objects_; tlr_idx++) {
				  int offset = tlr_idx * num_classes_;
				  CHECK(cur_data.tlr_cls_[tlr_idx] < num_classes_) << "TLR class is out of bound " << cur_data.tlr_cls_[tlr_idx] << ", " << cur_data.file_name_;
				  _tlr_cls[offset + cur_data.tlr_cls_[tlr_idx]] = Dtype(1);
				}*/
			}

			if (use_tlr_blobs_) {
				Dtype* _tlr_blobs = batch->tlr_blobs_.mutable_cpu_data() + batch->tlr_blobs_.offset(n);
				_tlr_blobs[0] = cur_data.tlr_blobs_;
			}

			if (use_tlr_blobReg_) {
				SetTLRRegData(cur_data.img_, cur_data.tlr_ct_pt_, _tlr_blobReg, mirror, width, height, resized_width, resized_height, crop_x, crop_y, im_scale_x, im_scale_y);
			}
		}

		if (!use_fixed_size_) {
			int max_width = 0, max_height = 0;
			for (int n = 0; n < batchsz_; n++) {
				if (max_width < vdata_[n]->shape(3)) {
					max_width = vdata_[n]->shape(3);
				}
				if (max_height < vdata_[n]->shape(2)) {
					max_height = vdata_[n]->shape(2);
				}
			}
			batch->data_.Reshape(batchsz_, input_channel_[input_type_], max_height, max_width);
			memset(batch->data_.mutable_cpu_data(), 0, batch->data_.count() * sizeof(Dtype));
			for (int i = 0; i < num_segments_; i++) {
				if (!segment_label_resize_[i]) {
					batch->seg_[i]->Reshape(batchsz_, 1 * (segment_map_scale_[i] * segment_map_scale_[i]), max_height / segment_map_scale_[i], max_width / segment_map_scale_[i]);
				}
				else {
					batch->seg_[i]->Reshape(batchsz_, 1, max_height / segment_map_scale_[i], max_width / segment_map_scale_[i]);
				}
				memset(batch->seg_[i]->mutable_cpu_data(), 0, batch->seg_[i]->count() * sizeof(Dtype));
			}
			if (use_edge_) {
				batch->edge_.Reshape(batchsz_, 1, max_height / segment_map_scale_[0], max_width / segment_map_scale_[0]);
				memset(batch->edge_.mutable_cpu_data(), 0, batch->edge_.count() * sizeof(Dtype));
			}
			for (int n = 0; n < batchsz_; n++) {
				for (int c = 0; c < vdata_[n]->shape(1); c++) {
					for (int h = 0; h < vdata_[n]->shape(2); h++) {
						memcpy(batch->data_.mutable_cpu_data() + batch->data_.offset(n, c, h), vdata_[n]->cpu_data() + vdata_[n]->offset(0, c, h), vdata_[n]->shape(3) * sizeof(Dtype));
					}
				}
				for (int i = 0; i < num_segments_; i++) {
					for (int c = 0; c < vseg_[n * num_segments_ + i]->shape(1); c++) {
						for (int h = 0; h < vseg_[n * num_segments_ + i]->shape(2); h++) {
							memcpy(batch->seg_[i]->mutable_cpu_data() + batch->seg_[i]->offset(n, c, h), vseg_[n * num_segments_ + i]->cpu_data() + vseg_[n * num_segments_ + i]->offset(0, c, h), vseg_[n * num_segments_ + i]->shape(3) * sizeof(Dtype));
						}
					}
					batch->seg_[i]->Reshape(batch->seg_[i]->shape(0), 1, batch->seg_[i]->shape(1) * batch->seg_[i]->shape(2), batch->seg_[i]->shape(3));
				}
				if (use_edge_) {
					for (int c = 0; c < vedge_[n]->shape(1); c++) {
						for (int h = 0; h < vedge_[n]->shape(2); h++) {
							memcpy(batch->edge_.mutable_cpu_data() + batch->edge_.offset(n, c, h), vedge_[n]->cpu_data() + vedge_[n]->offset(0, c, h), vedge_[n]->shape(3) * sizeof(Dtype));
						}
					}
				}
			}
		}

		if (DrawDebug::debug_mode) {
			for (int n = 0; n < batchsz_; n++) {
				const db::UDBDatum& cur_data = *data[n];
				cv::Mat image_ori = cur_data.img_.clone();

				int resized_width = batch->data_.shape(3);
				int resized_height = batch->data_.shape(2);
				cv::Mat image_trn(resized_height, resized_width, CV_8UC3, cv::Scalar(0));
				for (int i = 0; i < resized_height; i++) {
					for (int j = 0; j < resized_width; j++) {
						const Dtype* _data = batch->data_.cpu_data();
						image_trn.data[(i * resized_width + j) * 3 + 0] = _data[((n * 3 + 0) * resized_height + i) *  resized_width + j] + mean0_ + mean0_noise_;
						image_trn.data[(i * resized_width + j) * 3 + 1] = _data[((n * 3 + 1) * resized_height + i) *  resized_width + j] + mean1_ + mean1_noise_;
						image_trn.data[(i * resized_width + j) * 3 + 2] = _data[((n * 3 + 2) * resized_height + i) *  resized_width + j] + mean2_ + mean2_noise_;
					}
				}
				DrawDebug::SetImage(image_trn, n);

				for (int i = 0; i < cur_data.object_gt_.size(); i++) {
					const db::Box2D* box_2d = cur_data.object_gt_[i].get_box_2d(cur_data.box_2d_);
					const db::Quad2D* quad_2d = cur_data.object_gt_[i].get_quad_2d(cur_data.quad_2d_);
					const db::Box3D* box_3d = cur_data.object_gt_[i].get_box_3d(cur_data.box_3d_);
					const db::BoxNew3D* box_new_3d = cur_data.object_gt_[i].get_box_new_3d(cur_data.box_new_3d_);
					const db::BoxAttribute* box_attribute = cur_data.object_gt_[i].get_box_attribute(cur_data.box_attribute_);
					if (box_2d && (use_gt_ || use_dc_ || use_hn_ || use_hp_ || use_gt_ex_)) {
						DrawDebug::Draw2D(image_ori, box_2d->p, (int)box_2d->label);
						if (use_gt_quad_) {
							DrawDebug::DrawQuad(image_ori, quad_2d->p, (int)quad_2d->label);
						}
					}
					if (box_3d && (use_gt_3d_ || use_gt_ex_)) {
						DrawDebug::Draw3D(image_ori, box_3d->p, (int)box_3d->direction);
					}
					if (box_new_3d && (use_gt_new_3d_ || use_gt_ex_)) {
						DrawDebug::DrawNew3D(image_ori, box_new_3d->p, (int)box_new_3d->direction, (int)box_new_3d->shape);
					}
					if (box_2d && box_attribute && use_gt_ex_) {
						DrawDebug::DrawAttribute(image_ori, box_2d->x1, box_2d->y1, (int)box_attribute->direction, box_attribute->occluded, box_attribute->truncated, (int)box_attribute->age, (int)box_attribute->gender, (int)box_attribute->sit_stand);
					}
				}

				if (use_dc_) {
					for (int i = 0; i < batch->dc_rois_.shape(0); i++) {
						const Dtype* _dc_rois = batch->dc_rois_.cpu_data();
						if (_dc_rois[i * 5 + 0] == n) {
							DrawDebug::Draw2D(image_trn, (float*)&_dc_rois[i * 5 + 1], -100000);
						}
					}
				}
				if (use_hn_) {
					for (int i = 0; i < batch->hn_rois_.shape(0); i++) {
						const Dtype* _hn_rois = batch->hn_rois_.cpu_data();
						if (_hn_rois[i * 5 + 0] == n) {
							DrawDebug::Draw2D(image_trn, (float*)&_hn_rois[i * 5 + 1], -100001);
						}
					}
				}
				if (use_hp_) {
					for (int i = 0; i < batch->hp_rois_.shape(0); i++) {
						const Dtype* _hp_rois = batch->hp_rois_.cpu_data();
						if (_hp_rois[i * 5 + 0] == n) {
							DrawDebug::Draw2D(image_trn, (float*)&_hp_rois[i * 5 + 1], -100002);
						}
					}
				}
				if (use_gt_) {
					for (int i = 0; i < batch->gt_rois_.shape(0); i++) {
						const Dtype* _gt_rois = batch->gt_rois_.cpu_data();
						if (_gt_rois[i * 6 + 0] == n) {
							DrawDebug::Draw2D(image_trn, (float*)&_gt_rois[i * 6 + 1], (int)_gt_rois[i * 6 + 5]);
						}
						if (use_gt_quad_) {
							const Dtype* _gt_rois_quad = batch->gt_rois_quad_.cpu_data();
							if (_gt_rois_quad[i * 10 + 0] == n) {
								DrawDebug::DrawQuad(image_trn, (float*)&_gt_rois_quad[i * 10 + 1], (int)_gt_rois_quad[i * 10 + 9]);
							}
						}
					}
				}
				if (use_gt_3d_) {
					for (int i = 0; i < batch->gt_rois_3d_.shape(0); i++) {
						const Dtype* _gt_rois_3d = batch->gt_rois_3d_.cpu_data();
						if (_gt_rois_3d[i * 10 + 0] == n) {
							DrawDebug::Draw3D(image_trn, (float*)&_gt_rois_3d[i * 10 + 1], (int)_gt_rois_3d[i * 10 + 9]);
						}
					}
				}
				if (use_gt_new_3d_) {
					for (int i = 0; i < batch->gt_rois_new_3d_.shape(0); i++) {
						const Dtype* _gt_rois_new_3d = batch->gt_rois_new_3d_.cpu_data();
						if (_gt_rois_new_3d[i * 17 + 0] == n) {
							DrawDebug::DrawNew3D(image_trn, (float*)&_gt_rois_new_3d[i * 17 + 1], (int)_gt_rois_new_3d[i * 17 + 15], (int)_gt_rois_new_3d[i * 17 + 16]);
						}
					}
				}
				if (use_gt_ex_) {
					for (int i = 0; i < batch->gt_rois_ex_.shape(0); i++) {
						const Dtype* _gt_rois_ex = batch->gt_rois_ex_.cpu_data();
						if (_gt_rois_ex[i * 32 + 0] == n) {
							DrawDebug::Draw2D(image_trn, (float*)&_gt_rois_ex[i * 32 + 1], (int)_gt_rois_ex[i * 32 + 5]);
							if (post_type_ == POST_V1) {
								DrawDebug::DrawAttribute(image_trn, (float)_gt_rois_ex[i * 32 + 1], (float)_gt_rois_ex[i * 32 + 2], (int)_gt_rois_ex[i * 32 + 24], (float)_gt_rois_ex[i * 32 + 6], (float)_gt_rois_ex[i * 32 + 7]);
								if (_gt_rois_ex[i * 32 + 16] != -1)
									DrawDebug::Draw3D(image_trn, (float*)&_gt_rois_ex[i * 32 + 16], (int)_gt_rois_ex[i * 32 + 24]);
							}
							else if (post_type_ > POST_V1) {
								DrawDebug::DrawAttribute(image_trn, (float)_gt_rois_ex[i * 32 + 1], (float)_gt_rois_ex[i * 32 + 2], (int)_gt_rois_ex[i * 32 + 30], (float)_gt_rois_ex[i * 32 + 6], (float)_gt_rois_ex[i * 32 + 7], (int)_gt_rois_ex[i * 32 + 8], (int)_gt_rois_ex[i * 32 + 9], (int)_gt_rois_ex[i * 32 + 10]);
								if (_gt_rois_ex[i * 32 + 16] != -1)
									DrawDebug::DrawNew3D(image_trn, (float*)&_gt_rois_ex[i * 32 + 16], (int)_gt_rois_ex[i * 32 + 30], (int)_gt_rois_ex[i * 32 + 31]);
							}
						}
					}
				}
				cv::Mat ori;
				cv::Mat trn;
				if (use_seg_) {
					cv::Mat segmap_ori = cur_data.seg_.clone();
					int segmap_width = 0;
					for (int i = 0; i < num_segments_; i++) {
						segmap_width += segment_label_resize_[i] ? resized_width / segment_map_scale_[i] : resized_width;
					}
					cv::Mat segmap_trn(resized_height, segmap_width, CV_8UC3, cv::Scalar(0));
					for (int y = 0; y < resized_height; y++) {
						for (int x = 0; x < resized_width; x++) {
							int offset = 0;
							for (int i = 0; i < num_segments_; i++) {
								const Dtype* _seg = batch->seg_[i]->cpu_data();
								int seg_color = 0;
								int seg_c = (y % segment_map_scale_[i]) * segment_map_scale_[i] + (x % segment_map_scale_[i]);
								int seg_y = y / segment_map_scale_[i];
								int seg_x = x / segment_map_scale_[i];
								if (!segment_label_resize_[i]) {
									int seg_idx = _seg[((n * (segment_map_scale_[i] * segment_map_scale_[i]) + seg_c) * (resized_height / segment_map_scale_[i]) + seg_y) * (resized_width / segment_map_scale_[i]) + seg_x];
									seg_color = segment_map_inv_[i].find(seg_idx)->second;
									segmap_trn.data[(y * segmap_width + offset + x) * 3 + 0] = INT2B(seg_color);
									segmap_trn.data[(y * segmap_width + offset + x) * 3 + 1] = INT2G(seg_color);
									segmap_trn.data[(y * segmap_width + offset + x) * 3 + 2] = INT2R(seg_color);
								}
								else if (seg_c == 0) {
									if (seg_y < resized_height / segment_map_scale_[i] && seg_x < resized_width / segment_map_scale_[i]) {
										int seg_idx = _seg[(n * (resized_height / segment_map_scale_[i]) + seg_y) * (resized_width / segment_map_scale_[i]) + seg_x];
										seg_color = segment_map_inv_[i].find(seg_idx)->second;
									}
									segmap_trn.data[(seg_y * segmap_width + offset + seg_x) * 3 + 0] = INT2B(seg_color);
									segmap_trn.data[(seg_y * segmap_width + offset + seg_x) * 3 + 1] = INT2G(seg_color);
									segmap_trn.data[(seg_y * segmap_width + offset + seg_x) * 3 + 2] = INT2R(seg_color);
								}
								offset += segment_label_resize_[i] ? resized_width / segment_map_scale_[i] : resized_width;
							}
						}
					}
					if (use_edge_) {
						cv::Mat edgemap_trn(resized_height / segment_map_scale_[0], resized_width / segment_map_scale_[0], CV_8UC3, cv::Scalar(0));
						for (int y = 0; y < edgemap_trn.rows; y++) {
							for (int x = 0; x < edgemap_trn.cols; x++) {
								edgemap_trn.data[(y * edgemap_trn.cols + x) * 3 + 0] = batch->edge_.cpu_data()[(n * edgemap_trn.rows + y) * edgemap_trn.cols + x] * 255;
								edgemap_trn.data[(y * edgemap_trn.cols + x) * 3 + 1] = batch->edge_.cpu_data()[(n * edgemap_trn.rows + y) * edgemap_trn.cols + x] * 255;
								edgemap_trn.data[(y * edgemap_trn.cols + x) * 3 + 2] = batch->edge_.cpu_data()[(n * edgemap_trn.rows + y) * edgemap_trn.cols + x] * 255;
							}
						}
						trn = cv::Mat(image_trn.rows, image_trn.cols + segmap_trn.cols + edgemap_trn.cols, CV_8UC3, cv::Scalar(0));
						cv::Mat trn1 = trn(cv::Rect(0, 0, image_trn.cols, image_trn.rows));
						cv::Mat trn2 = trn(cv::Rect(image_trn.cols, 0, segmap_trn.cols, segmap_trn.rows));
						cv::Mat trn3 = trn(cv::Rect(image_trn.cols + segmap_trn.cols, 0, edgemap_trn.cols, edgemap_trn.rows));
						image_trn.copyTo(trn1);
						segmap_trn.copyTo(trn2);
						edgemap_trn.copyTo(trn3);
					}
					else {
						trn = cv::Mat(image_trn.rows, image_trn.cols + segmap_trn.cols, CV_8UC3, cv::Scalar(0));
						cv::Mat trn1 = trn(cv::Rect(0, 0, image_trn.cols, image_trn.rows));
						cv::Mat trn2 = trn(cv::Rect(image_trn.cols, 0, segmap_trn.cols, segmap_trn.rows));
						image_trn.copyTo(trn1);
						segmap_trn.copyTo(trn2);
					}
					ori = cv::Mat(image_ori.rows, image_ori.cols + segmap_ori.cols, CV_8UC3, cv::Scalar(0));
					cv::Mat ori1 = ori(cv::Rect(0, 0, image_ori.cols, image_ori.rows));
					cv::Mat ori2 = ori(cv::Rect(image_ori.cols, 0, segmap_ori.cols, segmap_ori.rows));
					image_ori.copyTo(ori1);
					segmap_ori.copyTo(ori2);
				}
				else {
					ori = image_ori;
					trn = image_trn;
				}
				//if (ori.cols > 1920) {
				//  float scale = 1920.f / ori.cols;
				//  cv::resize(ori, ori, cv::Size(), scale, scale);
				//}
				//if (trn.cols > 1920) {
				//  float scale = 1920.f / trn.cols;
				//  cv::resize(trn, trn, cv::Size(), scale, scale);
				//}
				char text[512];
				sprintf(text, "[%d]%s/%s", n + 1, cur_data.db_name_.c_str(), cur_data.file_name_.c_str());
				cv::putText(ori, text, cv::Point(10, 15), 2, 0.5, cv::Scalar(0, 0, 0), 3);
				cv::putText(ori, text, cv::Point(10, 15), 2, 0.5, cv::Scalar(255, 255, 255), 1);
				sprintf(text, "%dx%d", image_ori.cols, image_ori.rows);
				cv::putText(ori, text, cv::Point(10, 30), 2, 0.5, cv::Scalar(0, 0, 0), 3);
				cv::putText(ori, text, cv::Point(10, 30), 2, 0.5, cv::Scalar(255, 255, 255), 1);
				sprintf(text, "[%d]UDBDataLayer(%s/%s)", n + 1, cur_data.db_name_.c_str(), cur_data.file_name_.c_str());
				cv::putText(trn, text, cv::Point(10, 15), 2, 0.5, cv::Scalar(0, 0, 0), 3);
				cv::putText(trn, text, cv::Point(10, 15), 2, 0.5, cv::Scalar(255, 255, 255), 1);
				sprintf(text, "%dx%d", image_trn.cols, image_trn.rows);
				cv::putText(trn, text, cv::Point(10, 30), 2, 0.5, cv::Scalar(0, 0, 0), 3);
				cv::putText(trn, text, cv::Point(10, 30), 2, 0.5, cv::Scalar(255, 255, 255), 1);
				DrawDebug::ShowCanvas("Original Image", 0, n, true);
				DrawDebug::ShowCanvas("Training Image", 1, n, true, 2560, 1440);
				DrawDebug::ShowCanvas("Feature Image", 2, n, true, 2560, 1440);
				DrawDebug::AddImagetoCanvas(ori, 0, n);
				DrawDebug::AddImagetoCanvas(trn, 1, n);
				if (!DrawDebug::debug_mode)
					break;
			}
		}
		return true;
	}

	bool DrawDebug::debug_mode = false;
	int DrawDebug::wait_time = 0;
	std::vector<cv::Mat> DrawDebug::images;
	std::vector<std::vector<cv::Mat> > DrawDebug::canvas;
	std::vector<std::vector<int> > DrawDebug::image_cnt;

	//#ifdef CPU_ONLY
	//STUB_GPU(UDBDataLayer);
	//#endif

	INSTANTIATE_CLASS(UDBDataLayer);
	REGISTER_LAYER_CLASS(UDBData);

}  // namespace caffe
#endif  // USE_OPENCV
