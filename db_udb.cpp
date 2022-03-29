#ifdef USE_OPENCV
#include "caffe/util/db_udb.hpp"
#include "caffe/util/strparam.hpp"
#include "caffe/util/path_utils.hpp"
#include <opencv2/opencv.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

#define UDB_CACHE_VER "ucv1.1"

namespace caffe {
	namespace db {

		void UDBCursor::SeekToFirst() {
			if (shuffle_) {
				std::random_shuffle(shuffled_index_.begin(), shuffled_index_.end());
			}
			idx_ = 0;
			seq_idx_ = 0;
		}

		void UDBCursor::Next() {
			if (idx_ == shuffled_index_.size() - 1 && seq_idx_ == sequence_) {
				SeekToFirst();
			}
			else {
				if (seq_idx_ < sequence_) {
					seq_idx_++;
				}
				else {
					idx_++;
					seq_idx_ = 0;
				}
			}
		}

		static void make_db_files(std::string srcroot, std::string src, std::vector<std::pair<std::vector<std::string>, std::string> >& db_files) {
			boost::trim(src);
			size_t idx = src.rfind(":");
			CHECK(idx != std::string::npos) << "'source' in udb_data_param should follow this format - '[tar path]:[dataset name]'";
			std::vector<string> tar_files;
			std::string tar_token = src.substr(0, idx);
			size_t tar_idx;
			while ((tar_idx = tar_token.find(",")) != std::string::npos) {
				if (tar_token.substr(0, tar_idx) == "-") {
					tar_files.push_back("");
				}
				else {
					tar_files.push_back(srcroot + tar_token.substr(0, tar_idx) + ".tar");
				}
				tar_token = tar_token.substr(tar_idx + 1);
			}
			tar_files.push_back(srcroot + tar_token + ".tar");
			db_files.push_back(std::make_pair(tar_files, src.substr(idx + 1)));
		}

		static string class_index_to_string(int index) {
			if (index == 0) {
				return "background";
			}
			else if (index == -100000) {
				return "dont_care";
			}
			else if (index == -100001) {
				return "hard_negative";
			}
			else {
				return string("class(") + boost::lexical_cast<string>(index) + string(")");
			}
		}

		UDB::UDB(const UDBDataParameter& param, const int src_index, const char* cache_file, bool cache_write)
			: param_(param), sequence_(param.sequence()) {
			if (src_index == -100001) {
				CHECK(param_.false_positive_list().size() > 0);
				for (int i = 0; i < param_.false_positive_list().size(); i++) {
					std::ifstream f(param_.false_positive_list(i).c_str());
					std::string line;
					while (getline(f, line)) {
						make_db_files(param_.source_root(), line, db_files_);
					}
				}
			}
			else {
				// parse source parameter in udb_data_param
				CHECK(param_.source_list().size() > 0);
				std::ifstream f(param_.source_list(src_index).c_str());
				std::string line;
				while (getline(f, line)) {
					make_db_files(param_.source_root(), line, db_files_);
				}
				if (param_.use_freq()) {
					std::string freq_file = param_.source_list(src_index) + ".freq";
					std::ifstream f_freq(freq_file.c_str());
					if (f_freq.is_open()) {
						LOG(INFO) << "Loading... " << freq_file;
						while (getline(f_freq, line)) {
							boost::trim(line);
							size_t idx = line.find(",");
							freq_[line.substr(0, idx)] = atoi(line.substr(idx + 1).c_str());
						}
					}
				}
			}

			CHECK(db_files_.size() > 0) << "There is no readable sources";

			use_img_ = false;
			use_od_ = false;
			use_od_quad_ = false;
			use_3d_ = false;
			use_new_3d_ = false;
			use_od_ex_ = false;
			use_seg_ = false;
			use_scene_ = false;
			use_od_dontcare_ = false;
			use_od_hard_negative_ = false;
			use_failsafe_ = false;
			use_ego_out_ = false;
			use_tsr_cls_ = false;
			use_tlr_cls_ = false;
			use_tlr_blob_ = false;
			use_tlr_blobReg_ = false;
			use_meta_info_ = false;
			use_lane_type_label_ = false;
			use_boundary_type_label_ = false;

			for (int i = 0; i < param_.top_size(); i++) {
				if (param_.top(i) == UDBDataParameter_TopConfiguration_IMG)
					use_img_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_GT_ROIS)
					use_od_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_GT_ROIS_QUAD)
					use_od_quad_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_GT_ROIS_3D)
					use_3d_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_GT_ROIS_NEW_3D)
					use_new_3d_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_GT_ROIS_EX)
					use_od_ex_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_SEG)
					use_seg_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_SCENE_LABEL)
					use_scene_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_DC_ROIS)
					use_od_dontcare_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_HN_ROIS)
					use_od_hard_negative_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_FAILSAFE)
					use_failsafe_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_EGO_OUT)
					use_ego_out_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_CLOSENESS)
					use_od_ = true; // od annotation is required to get closeness
				if (param_.top(i) == UDBDataParameter_TopConfiguration_TSR_CLS)
					use_tsr_cls_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_TLR_CLS)
					use_tlr_cls_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_TLR_BLOBS)
					use_tlr_blob_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_TLR_BLOBREG)
					use_tlr_blobReg_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_META_INFO)
					use_meta_info_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_LANE_TYPE_LABEL)
					use_lane_type_label_ = true;
				if (param_.top(i) == UDBDataParameter_TopConfiguration_BOUNDARY_TYPE_LABEL)
					use_boundary_type_label_ = true;
			}
			CHECK(param_.source_info_size() == 0 || param_.source_info_size() == param_.source_list_size());
			if (src_index == -100001) {
				use_od_ = false;
				use_od_quad_ = false;
				use_3d_ = false;
				use_new_3d_ = false;
				use_od_ex_ = false;
				use_seg_ = false;
				use_scene_ = false;
				use_od_dontcare_ = false;
				use_od_hard_negative_ = false;
				use_failsafe_ = false;
				use_ego_out_ = false;
				use_tsr_cls_ = false;
				use_tlr_cls_ = false;
				use_tlr_blob_ = false;
				use_tlr_blobReg_ = false;
				use_meta_info_ = false;
				req_od_ = use_od_;
				req_od_quad_ = use_od_quad_;
				req_3d_ = use_3d_;
				req_new_3d_ = use_new_3d_;
				req_od_ex_ = use_od_ex_;
				req_seg_ = use_seg_;
				req_scene_ = use_scene_;
				req_tsr_cls_ = use_tsr_cls_;
				req_tlr_cls_ = use_tlr_cls_;
				req_tlr_blob_ = use_tlr_blob_;
				req_tlr_blobReg_ = use_tlr_blobReg_;
				req_meta_info_ = use_meta_info_;
			}
			else if (param_.source_info_size() > 0) {
				std::vector<string> req_info;
				parse_strparam_s(param_.source_info(src_index).c_str(), req_info);
				req_od_ = false;
				req_od_quad_ = false;
				req_3d_ = false;
				req_new_3d_ = false;
				req_od_ex_ = false;
				req_seg_ = false;
				req_scene_ = false;
				req_tsr_cls_ = false;
				req_tlr_cls_ = false;
				req_tlr_blob_ = false;
				req_tlr_blobReg_ = false;
				req_meta_info_ = false;
				for (int i = 0; i < req_info.size(); i++) {
					if (strcmp(req_info[i].c_str(), "req_od") == 0)
						req_od_ = true;
					if (strcmp(req_info[i].c_str(), "req_od_quad") == 0)
						req_od_quad_ = true;
					if (strcmp(req_info[i].c_str(), "req_3d") == 0)
						req_3d_ = true;
					if (strcmp(req_info[i].c_str(), "req_new_3d") == 0)
						req_new_3d_ = true;
					if (strcmp(req_info[i].c_str(), "req_od_ex") == 0)
						req_od_ex_ = true;
					if (strcmp(req_info[i].c_str(), "req_seg") == 0)
						req_seg_ = true;
					if (strcmp(req_info[i].c_str(), "req_scene") == 0)
						req_scene_ = true;
					if (strcmp(req_info[i].c_str(), "req_tsr_cls") == 0)
						req_tsr_cls_ = true;
					if (strcmp(req_info[i].c_str(), "req_tlr_cls") == 0)
						req_tlr_cls_ = true;
					if (strcmp(req_info[i].c_str(), "req_tlr_blob") == 0)
						req_tlr_blob_ = true;
					if (strcmp(req_info[i].c_str(), "req_tlr_blobReg") == 0)
						req_tlr_blobReg_ = true;
					if (strcmp(req_info[i].c_str(), "req_meta_info") == 0)
						req_meta_info_ = true;
				}
			}
			else {
				req_od_ = use_od_;
				req_od_quad_ = use_od_quad_;
				req_3d_ = use_3d_;
				req_new_3d_ = use_new_3d_;
				req_od_ex_ = use_od_ex_;
				req_seg_ = use_seg_;
				req_scene_ = use_scene_;
				req_tsr_cls_ = use_tsr_cls_;
				req_tlr_cls_ = use_tlr_cls_;
				req_tlr_blob_ = use_tlr_blob_;
				req_tlr_blobReg_ = use_tlr_blobReg_;
				req_meta_info_ = use_meta_info_;
			}

			if (param_.class_string().size()) {
				std::vector<std::vector<string> > res;
				parse_strparam_s(param_.class_string().c_str(), res);
				CHECK(res.size() > 0);

				clslst_.resize(res[0].size());
				clslst_.assign(res[0].begin(), res[0].end());
			}

			if (param_.class_index().size()) {
				std::vector<std::vector<int> > res;
				parse_strparam_i(param_.class_index().c_str(), res);
				CHECK(res.size() > 0);
				clsidxlst_.resize(res[0].size());
				clsidxlst_.assign(res[0].begin(), res[0].end());
			}

			if (src_index < param_.mandatory_class_size()) {
				parse_strparam_i(param_.mandatory_class(src_index).c_str(), mandatory_class_);
			}

			cursor_ = NULL;

			use_cache_ = param_.use_cache() && cache_file && cache_file[0];
			cache_write_ = use_cache_ && cache_write;
			if (use_cache_) {
				strcpy(cache_file_, cache_file);
				strcpy(cache_temp_file_, cache_file);
				strcat(cache_temp_file_, ".tmp");
			}
		}

		static std::string get_name(ptree::value_type const& v) {
			std::string clsname = "";
			try {
				clsname = v.second.get<std::string>("name");
				boost::trim(clsname);
				try {
					if (v.second.get<int>("vehicle_size")) {
						clsname += "_wide";
					}
					else {
						clsname += "_normal";
					}
				}
				catch (std::exception const& e) {
				}
				try {
					if (!v.second.get<int>("with_human")) {
						clsname += "_wo_human";
					}
				}
				catch (std::exception const& e) {
				}
			}
			catch (std::exception const& e) {
				clsname = "";
			}
			return clsname;
		}

		static int get_difficult(ptree::value_type const& v) {
			int difficult = 0;
			try {
				return v.second.get<int>("difficult");
			}
			catch (std::exception const& e) {
			}
			return 0;
		}

		static int get_ambiguity(ptree::value_type const& v) {
			try {
				return v.second.get<int>("ambiguity");
			}
			catch (std::exception const& e) {
			}
			return 0;
		}

		static int get_irrelevant(ptree::value_type const& v) {
			try {
				return v.second.get<int>("evaluated");
			}
			catch (std::exception const& e) {
			}
			return 0;
		}

		static float get_occluded(ptree::value_type const& v) {
			try {
				return v.second.get<float>("occluded");
			}
			catch (std::exception const& e) {
			}
			return -1;
		}

		static float get_truncated(ptree::value_type const& v) {
			try {
				return v.second.get<float>("truncated");
			}
			catch (std::exception const& e) {
			}
			return -1;
		}

		static int get_sit_stand(ptree::value_type const& v) {
			try {
				return v.second.get<int>("sit_stand");
			}
			catch (std::exception const& e) {
			}
			return -1;
		}

		static int get_age(ptree::value_type const& v) {
			try {
				return v.second.get<int>("age");
			}
			catch (std::exception const& e) {
			}
			return -1;
		}

		static int get_gender(ptree::value_type const& v) {
			try {
				return v.second.get<int>("gender");
			}
			catch (std::exception const& e) {
			}
			return -1;
		}

		static int get_direction(ptree::value_type const& v) {
			try {
				return v.second.get<int>("direction");
			}
			catch (std::exception const& e) {
			}
			return -1;
		}

		static bool get_bbox(ptree::value_type const& v, const int width, const int height, const std::string tar_path, const std::string ann_path, float& x1, float& y1, float& x2, float& y2) {
			try {
				const ptree& bbox_ent = v.second.get_child("bndbox");
				if (strstr(tar_path.c_str(), "VOC")) {
					x1 = std::max<float>(std::min<float>(bbox_ent.get<float>("xmin") - 1, width - 1), 0);
					y1 = std::max<float>(std::min<float>(bbox_ent.get<float>("ymin") - 1, height - 1), 0);
				}
				else {
					x1 = std::max<float>(std::min<float>(bbox_ent.get<float>("xmin"), width - 1), 0);
					y1 = std::max<float>(std::min<float>(bbox_ent.get<float>("ymin"), height - 1), 0);
				}
				x2 = std::max<float>(std::min<float>(bbox_ent.get<float>("xmax") - 1, width - 1), 0);
				y2 = std::max<float>(std::min<float>(bbox_ent.get<float>("ymax") - 1, height - 1), 0);
				if (x1 >= x2) {
					LOG(INFO) << tar_path << "/" << ann_path << ": xmin(" << x1 << ") >= xmax(" << x2 << ")" << std::endl;
				}
				else if (y1 >= y2) {
					LOG(INFO) << tar_path << "/" << ann_path << ": ymin(" << y1 << ") >= ymax(" << y2 << ")" << std::endl;
				}
				else {
					return true;
				}
			}
			catch (std::exception const& e) {
			}
			return false;
		}

		static bool get_quad(ptree::value_type const& v, const int width, const int height, const std::string tar_path, const std::string ann_path, float& x1, float& y1, float& x2, float& y2, float& x3, float& y3, float& x4, float& y4) {
			try {
				const ptree& quad_ent = v.second.get_child("bndbox");

				//x1 = std::max<float>(std::min<float>(v.second.get<float>("x1"), width - 1), 0);
				//y1 = std::max<float>(std::min<float>(v.second.get<float>("y1"), height - 1), 0);
				//x2 = std::max<float>(std::min<float>(v.second.get<float>("x2"), width - 1), 0);
				//y2 = std::max<float>(std::min<float>(v.second.get<float>("y2"), height - 1), 0);
				//x3 = std::max<float>(std::min<float>(v.second.get<float>("x3"), width - 1), 0);
				//y3 = std::max<float>(std::min<float>(v.second.get<float>("y3"), height - 1), 0);
				//x4 = std::max<float>(std::min<float>(v.second.get<float>("x4"), width - 1), 0);
				//y4 = std::max<float>(std::min<float>(v.second.get<float>("y4"), height - 1), 0);

				x1 = std::max<float>(std::min<float>(quad_ent.get<float>("<xmlattr>.x1"), width - 1), 0);
				y1 = std::max<float>(std::min<float>(quad_ent.get<float>("<xmlattr>.y1"), height - 1), 0);
				x2 = std::max<float>(std::min<float>(quad_ent.get<float>("<xmlattr>.x2"), width - 1), 0);
				y2 = std::max<float>(std::min<float>(quad_ent.get<float>("<xmlattr>.y2"), height - 1), 0);
				x3 = std::max<float>(std::min<float>(quad_ent.get<float>("<xmlattr>.x3"), width - 1), 0);
				y3 = std::max<float>(std::min<float>(quad_ent.get<float>("<xmlattr>.y3"), height - 1), 0);
				x4 = std::max<float>(std::min<float>(quad_ent.get<float>("<xmlattr>.x4"), width - 1), 0);
				y4 = std::max<float>(std::min<float>(quad_ent.get<float>("<xmlattr>.y4"), height - 1), 0);

				return true;

			}
			catch (std::exception const& e) {
			}
			return false;
		}

		static bool get_bbox_3d(ptree::value_type const& v, const int width, const int height, const std::string tar_path, const std::string ann_path, float& x1, float& y1, float& x2, float& y2, float& x3, float& y3, float& x4, float& y4) {
			try {
				const ptree& bbox_ent = v.second.get_child("hexahedron");
				x1 = std::max<float>(std::min<float>(bbox_ent.get<float>("x1"), width - -1), 0);
				y1 = std::max<float>(std::min<float>(bbox_ent.get<float>("y1"), height - 1), 0);
				x2 = std::max<float>(std::min<float>(bbox_ent.get<float>("x2") - 1, width - 1), 0);
				y2 = std::max<float>(std::min<float>(bbox_ent.get<float>("y2") - 1, height - 1), 0);
				x3 = std::max<float>(std::min<float>(bbox_ent.get<float>("x3"), width - -1), 0);
				y3 = std::max<float>(std::min<float>(bbox_ent.get<float>("y3"), height - 1), 0);
				x4 = std::max<float>(std::min<float>(bbox_ent.get<float>("x4") - 1, width - 1), 0);
				y4 = std::max<float>(std::min<float>(bbox_ent.get<float>("y4") - 1, height - 1), 0);

				x1 = std::max<float>(std::min<float>(x1, x2 - 1), 0);
				y1 = std::max<float>(std::min<float>(y1, y2 - 1), 0);
				x2 = std::max<float>(std::min<float>(x2, width - 1), x1 + 1);
				y2 = std::max<float>(std::min<float>(y2, height - 1), y1 + 1);
				x3 = std::max<float>(std::min<float>(x3, x4 - 1), 0);
				y3 = std::max<float>(std::min<float>(y3, y4 - 1), 0);
				x4 = std::max<float>(std::min<float>(x4, width - 1), x3 + 1);
				y4 = std::max<float>(std::min<float>(y4, height - 1), y3 + 1);

				x1 = std::max<float>(std::min<float>(x1, width - 1), 0);
				y1 = std::max<float>(std::min<float>(y1, height - 1), 0);
				x2 = std::max<float>(std::min<float>(x2, width - 1), 0);
				y2 = std::max<float>(std::min<float>(y2, height - 1), 0);
				x3 = std::max<float>(std::min<float>(x3, width - 1), 0);
				y3 = std::max<float>(std::min<float>(y3, height - 1), 0);
				x4 = std::max<float>(std::min<float>(x4, width - 1), 0);
				y4 = std::max<float>(std::min<float>(y4, height - 1), 0);
				if (x1 > x2) {
					LOG(INFO) << tar_path << "/" << ann_path << ": x1(" << x1 << ") > x2(" << x2 << ")" << std::endl;
				}
				else if (y1 > y2) {
					LOG(INFO) << tar_path << "/" << ann_path << ": y1(" << y1 << ") > y2(" << y2 << ")" << std::endl;
				}
				else if (x3 > x4) {
					LOG(INFO) << tar_path << "/" << ann_path << ": x3(" << x3 << ") > x4(" << x4 << ")" << std::endl;
				}
				else if (y3 > y4) {
					LOG(INFO) << tar_path << "/" << ann_path << ": y3(" << y3 << ") > y4(" << y4 << ")" << std::endl;
				}
				else {
					return true;
				}
			}
			catch (std::exception const& e) {
			}
			return false;
		}

		bool UDB::get_od_data(ptree::value_type const& v, const int width, const int height, const std::string& tar_path, const std::string& ann_path, std::string& clsname, Box2D& box_2d, BoxAttribute& box_attribute) {
			clsname = get_name(v);
			if (clsname != "") {
				box_2d.label = from_clsname(clsname);
				if (box_2d.label == INT_MAX) {
					LOG(ERROR) << "unrecognized class name " << clsname << " for " << ann_path << std::endl;
					exit(-1);
				}
				int difficult = get_difficult(v);
				int ambiguity = get_ambiguity(v);
				int irrelevant = get_irrelevant(v);
				box_attribute.direction = get_direction(v);
				box_attribute.occluded = get_occluded(v);
				box_attribute.truncated = get_truncated(v);
				box_attribute.sit_stand = get_sit_stand(v);
				box_attribute.age = get_age(v);
				box_attribute.gender = get_gender(v);
				if (!use_od_dontcare_ && box_2d.label == -100000)
					box_2d.label = 0;
				if (!use_od_hard_negative_ && box_2d.label == -100001)
					box_2d.label = 0;
				if (!param_.use_difficult() && difficult && box_2d.label > 0)
					box_2d.label *= -1;
				if (param_.ignore_ambiguous_positive() && ambiguity && box_2d.label > 0)
					box_2d.label *= -1;
				if (param_.ignore_irrelevant_positive() && irrelevant && box_2d.label > 0)
					box_2d.label *= -1;
				if (param_.ignore_occluded_positive() && (box_attribute.occluded == -1 || box_attribute.occluded > param_.ignore_occlusion_thresh()) && box_2d.label > 0)
					box_2d.label *= -1;
				if (param_.ignore_truncated_positive() && (box_attribute.truncated == -1 || box_attribute.truncated > param_.ignore_truncation_thresh()) && box_2d.label > 0)
					box_2d.label *= -1;
				if (box_2d.label != 0) {
					return get_bbox(v, width, height, tar_path, ann_path, box_2d.x1, box_2d.y1, box_2d.x2, box_2d.y2);
				}
			}
			return false;
		}

		bool UDB::get_quad_data(ptree::value_type const& v, const int width, const int height, const std::string& tar_path, const std::string& ann_path, std::string& clsname, Box2D& box_2d, Quad2D& quad_2d, BoxAttribute& box_attribute) {
			clsname = get_name(v);
			if (clsname != "") {
				quad_2d.label = from_clsname(clsname);
				if (quad_2d.label == INT_MAX) {
					LOG(ERROR) << "unrecognized class name " << clsname << " for " << ann_path << std::endl;
					exit(-1);
				}
				int difficult = get_difficult(v);
				int ambiguity = get_ambiguity(v);
				int irrelevant = get_irrelevant(v);
				box_attribute.direction = get_direction(v);
				box_attribute.occluded = get_occluded(v);
				box_attribute.truncated = get_truncated(v);
				box_attribute.sit_stand = get_sit_stand(v);
				box_attribute.age = get_age(v);
				box_attribute.gender = get_gender(v);
				if (!use_od_dontcare_ && quad_2d.label == -100000)
					quad_2d.label = 0;
				if (!use_od_hard_negative_ && quad_2d.label == -100001)
					quad_2d.label = 0;
				if (!param_.use_difficult() && difficult && quad_2d.label > 0)
					quad_2d.label *= -1;
				if (param_.ignore_ambiguous_positive() && ambiguity && quad_2d.label > 0)
					quad_2d.label *= -1;
				if (param_.ignore_irrelevant_positive() && irrelevant && quad_2d.label > 0)
					quad_2d.label *= -1;
				if (param_.ignore_occluded_positive() && box_attribute.occluded > 0 && quad_2d.label > 0)
					quad_2d.label *= -1;
				if (param_.ignore_truncated_positive() && box_attribute.truncated > 0 && quad_2d.label > 0)
					quad_2d.label *= -1;
				if (quad_2d.label != 0) {
					if (get_quad(v, width, height, tar_path, ann_path, quad_2d.x1, quad_2d.y1, quad_2d.x2, quad_2d.y2, quad_2d.x3, quad_2d.y3, quad_2d.x4, quad_2d.y4)) {
						box_2d.label = quad_2d.label;
						box_2d.x1 = std::min<float>(std::min<float>(quad_2d.x1, quad_2d.x2), std::min<float>(quad_2d.x3, quad_2d.x4));
						box_2d.y1 = std::min<float>(std::min<float>(quad_2d.y1, quad_2d.y2), std::min<float>(quad_2d.y3, quad_2d.y4));
						box_2d.x2 = std::max<float>(std::max<float>(quad_2d.x1, quad_2d.x2), std::max<float>(quad_2d.x3, quad_2d.x4));
						box_2d.y2 = std::max<float>(std::max<float>(quad_2d.y1, quad_2d.y2), std::max<float>(quad_2d.y3, quad_2d.y4));
						return true;
					}
				}
			}
			return false;
		}


		bool UDB::get_3d_data(ptree::value_type const& v, const int width, const int height, const std::string& tar_path, const std::string& ann_path, Box3D& box_3d) {
			if (!get_bbox_3d(v, width, height, tar_path, ann_path, box_3d.x1, box_3d.y1, box_3d.x2, box_3d.y2, box_3d.x3, box_3d.y3, box_3d.x4, box_3d.y4) ||
				(box_3d.direction = get_direction(v)) == -1) {
				return false;
			}
			if (box_3d.direction < 1 || box_3d.direction > 8) {
				LOG(ERROR) << "unrecognized direction " << box_3d.direction << " for " << ann_path << std::endl;
				exit(-1);
			}
			return true;
		}

		bool UDB::get_new_3d_data(ptree::value_type const& v, const int width, const int height, const std::string& tar_path, const std::string& ann_path, BoxNew3D& box_new_3d) {
			try {
				box_new_3d.direction = v.second.get<int>("<xmlattr>.Direction");
				if (box_new_3d.direction < 1 || box_new_3d.direction > 9) {
					LOG(ERROR) << "unrecognized direction " << box_new_3d.direction << " for " << ann_path << std::endl;
					exit(-1);
				}
				box_new_3d.shape = v.second.get<int>("<xmlattr>.Shape");
				if (box_new_3d.shape < 0 || box_new_3d.shape > 5) {
					LOG(ERROR) << "unrecognized shape " << box_new_3d.shape << " for " << ann_path << std::endl;
					exit(-1);
				}
				int cnt = 0;
				BOOST_FOREACH(ptree::value_type const & vv, v.second) {
					if (vv.first == "Point") {
						box_new_3d.p[cnt * 2 + 0] = std::max<float>(std::min<float>(vv.second.get<float>("<xmlattr>.x"), width - 1), 0);
						box_new_3d.p[cnt * 2 + 1] = std::max<float>(std::min<float>(vv.second.get<float>("<xmlattr>.y"), height - 1), 0);
						cnt++;
					}
				}
				if ((box_new_3d.shape == 0 && cnt != 4) ||
					(box_new_3d.shape >= 1 && box_new_3d.shape <= 3 && cnt != 6) ||
					(box_new_3d.shape >= 4 && box_new_3d.shape <= 5 && cnt != 7)) {
					LOG(ERROR) << "invalid number of points in shape (" << box_new_3d.shape << ")  for " << ann_path << std::endl;
					exit(-1);
				}
				box_new_3d.x1 = std::max<float>(std::min<float>(box_new_3d.x1, box_new_3d.x2 - 1), 0);
				box_new_3d.x2 = std::max<float>(std::min<float>(box_new_3d.x2, width - 1), box_new_3d.x1 + 1);
				box_new_3d.x3 = std::max<float>(std::min<float>(box_new_3d.x3, box_new_3d.x4 - 1), 0);
				box_new_3d.x4 = std::max<float>(std::min<float>(box_new_3d.x4, width - 1), box_new_3d.x3 + 1);
				box_new_3d.y1 = std::max<float>(std::min<float>(box_new_3d.y1, box_new_3d.y3 - 1), 0);
				box_new_3d.y2 = std::max<float>(std::min<float>(box_new_3d.y2, box_new_3d.y4 - 1), 0);
				box_new_3d.y3 = std::max<float>(std::min<float>(box_new_3d.y3, height - 1), box_new_3d.y1 + 1);
				box_new_3d.y4 = std::max<float>(std::min<float>(box_new_3d.y4, height - 1), box_new_3d.y2 + 1);
				if (box_new_3d.shape == 0) {
					for (int i = 0; i < 4; i++) {
						box_new_3d.p[i * 2 + 0] = std::max<float>(std::min<float>(box_new_3d.p[i * 2 + 0], width - 1), 0);
						box_new_3d.p[i * 2 + 1] = std::max<float>(std::min<float>(box_new_3d.p[i * 2 + 1], height - 1), 0);
					}
				}
				else if (box_new_3d.shape == 1) {
					box_new_3d.x5 = std::max<float>(std::min<float>(box_new_3d.x5, box_new_3d.x1 - 1), 0);
					box_new_3d.x6 = std::max<float>(std::min<float>(box_new_3d.x6, box_new_3d.x3 - 1), 0);
					box_new_3d.y5 = std::max<float>(std::min<float>(box_new_3d.y5, box_new_3d.y6 - 1), 0);
					box_new_3d.y6 = std::max<float>(std::min<float>(box_new_3d.y6, height - 1), box_new_3d.y5 + 1);
					for (int i = 0; i < 6; i++) {
						box_new_3d.p[i * 2 + 0] = std::max<float>(std::min<float>(box_new_3d.p[i * 2 + 0], width - 1), 0);
						box_new_3d.p[i * 2 + 1] = std::max<float>(std::min<float>(box_new_3d.p[i * 2 + 1], height - 1), 0);
					}
				}
				else if (box_new_3d.shape == 2) {
					box_new_3d.x5 = std::max<float>(std::min<float>(box_new_3d.x5, width - 1), box_new_3d.x2 + 1);
					box_new_3d.x6 = std::max<float>(std::min<float>(box_new_3d.x6, width - 1), box_new_3d.x4 + 1);
					box_new_3d.y5 = std::max<float>(std::min<float>(box_new_3d.y5, box_new_3d.y6 - 1), 0);
					box_new_3d.y6 = std::max<float>(std::min<float>(box_new_3d.y6, height - 1), box_new_3d.y5 + 1);
					for (int i = 0; i < 6; i++) {
						box_new_3d.p[i * 2 + 0] = std::max<float>(std::min<float>(box_new_3d.p[i * 2 + 0], width - 1), 0);
						box_new_3d.p[i * 2 + 1] = std::max<float>(std::min<float>(box_new_3d.p[i * 2 + 1], height - 1), 0);
					}
				}
				else if (box_new_3d.shape == 3) {
					box_new_3d.x5 = std::max<float>(std::min<float>(box_new_3d.x5, box_new_3d.x6 - 1), 0);
					box_new_3d.x6 = std::max<float>(std::min<float>(box_new_3d.x6, width - 1), box_new_3d.x5 + 1);
					box_new_3d.y5 = std::max<float>(std::min<float>(box_new_3d.y5, box_new_3d.y1 - 1), 0);
					box_new_3d.y6 = std::max<float>(std::min<float>(box_new_3d.y6, box_new_3d.y2 - 1), 0);
					for (int i = 0; i < 6; i++) {
						box_new_3d.p[i * 2 + 0] = std::max<float>(std::min<float>(box_new_3d.p[i * 2 + 0], width - 1), 0);
						box_new_3d.p[i * 2 + 1] = std::max<float>(std::min<float>(box_new_3d.p[i * 2 + 1], height - 1), 0);
					}
				}
				else if (box_new_3d.shape == 4) {
					box_new_3d.x5 = std::max<float>(std::min<float>(box_new_3d.x5, box_new_3d.x1 - 1), 0);
					box_new_3d.x6 = std::max<float>(std::min<float>(box_new_3d.x6, box_new_3d.x3 - 1), 0);
					box_new_3d.y5 = std::max<float>(std::min<float>(box_new_3d.y5, box_new_3d.y1 - 1), 0);
					box_new_3d.y6 = std::max<float>(std::min<float>(box_new_3d.y6, box_new_3d.y3 - 1), box_new_3d.y5 + 1);
					box_new_3d.x7 = std::max<float>(std::min<float>(box_new_3d.x7, box_new_3d.x2 - 1), box_new_3d.x5 + 1);
					box_new_3d.y7 = std::max<float>(std::min<float>(box_new_3d.y7, box_new_3d.y2 - 1), 0);
					for (int i = 0; i < 7; i++) {
						box_new_3d.p[i * 2 + 0] = std::max<float>(std::min<float>(box_new_3d.p[i * 2 + 0], width - 1), 0);
						box_new_3d.p[i * 2 + 1] = std::max<float>(std::min<float>(box_new_3d.p[i * 2 + 1], height - 1), 0);
					}
				}
				else if (box_new_3d.shape == 5) {
					box_new_3d.x5 = std::max<float>(std::min<float>(box_new_3d.x5, width - 1), box_new_3d.x2 + 1);
					box_new_3d.x6 = std::max<float>(std::min<float>(box_new_3d.x6, width - 1), box_new_3d.x4 + 1);
					box_new_3d.y5 = std::max<float>(std::min<float>(box_new_3d.y5, box_new_3d.y2 - 1), 0);
					box_new_3d.y6 = std::max<float>(std::min<float>(box_new_3d.y6, box_new_3d.y4 - 1), box_new_3d.y5 + 1);
					box_new_3d.x7 = std::max<float>(std::min<float>(box_new_3d.x7, box_new_3d.x5 - 1), box_new_3d.x1 + 1);
					box_new_3d.y7 = std::max<float>(std::min<float>(box_new_3d.y7, box_new_3d.y1 - 1), 0);
					for (int i = 0; i < 7; i++) {
						box_new_3d.p[i * 2 + 0] = std::max<float>(std::min<float>(box_new_3d.p[i * 2 + 0], width - 1), 0);
						box_new_3d.p[i * 2 + 1] = std::max<float>(std::min<float>(box_new_3d.p[i * 2 + 1], height - 1), 0);
					}
				}
				if (box_new_3d.x1 > box_new_3d.x2) {
					LOG(INFO) << tar_path << "/" << ann_path << ": pt1.x(" << box_new_3d.x1 << ") > pt2.x(" << box_new_3d.x2 << ")" << std::endl;
				}
				else if (box_new_3d.x3 > box_new_3d.x4) {
					LOG(INFO) << tar_path << "/" << ann_path << ": pt3.x(" << box_new_3d.x3 << ") > pt4.x(" << box_new_3d.x4 << ")" << std::endl;
				}
				else if (box_new_3d.y1 >= box_new_3d.y3) {
					LOG(INFO) << tar_path << "/" << ann_path << ": pt1.y(" << box_new_3d.y1 << ") >= pt3.y(" << box_new_3d.y3 << ")" << std::endl;
				}
				else if (box_new_3d.y2 >= box_new_3d.y4) {
					LOG(INFO) << tar_path << "/" << ann_path << ": pt2.y(" << box_new_3d.y2 << ") >= pt4.y(" << box_new_3d.y4 << ")" << std::endl;
				}
				else if (box_new_3d.shape == 3 && box_new_3d.x5 >= box_new_3d.x6) {
					LOG(INFO) << tar_path << "/" << ann_path << ": pt5.x(" << box_new_3d.x5 << ") > pt6.x(" << box_new_3d.x6 << ")" << std::endl;
				}
				else if (box_new_3d.shape == 4 && box_new_3d.x5 >= box_new_3d.x7) {
					LOG(INFO) << tar_path << "/" << ann_path << ": pt5.x(" << box_new_3d.x5 << ") > pt7.x(" << box_new_3d.x7 << ")" << std::endl;
				}
				else if (box_new_3d.shape == 5 && box_new_3d.x7 >= box_new_3d.x5) {
					LOG(INFO) << tar_path << "/" << ann_path << ": pt7.x(" << box_new_3d.x7 << ") > pt5.x(" << box_new_3d.x5 << ")" << std::endl;
				}
				else if ((box_new_3d.shape == 1 || box_new_3d.shape == 2 || box_new_3d.shape == 4 || box_new_3d.shape == 5) && box_new_3d.y5 > box_new_3d.y6) {
					LOG(INFO) << tar_path << "/" << ann_path << ": pt5.y(" << box_new_3d.y5 << ") > pt6.y(" << box_new_3d.y6 << ")" << std::endl;
				}
				else if (box_new_3d.shape == 3 && box_new_3d.y5 > box_new_3d.y1) {
					LOG(INFO) << tar_path << "/" << ann_path << ": pt5.y(" << box_new_3d.y5 << ") > pt1.y(" << box_new_3d.y1 << ")" << std::endl;
				}
				else if (box_new_3d.shape == 3 && box_new_3d.y6 > box_new_3d.y2) {
					LOG(INFO) << tar_path << "/" << ann_path << ": pt6.y(" << box_new_3d.y6 << ") > pt2.y(" << box_new_3d.y2 << ")" << std::endl;
				}
				else if (box_new_3d.shape == 4 && box_new_3d.y5 > box_new_3d.y1) {
					LOG(INFO) << tar_path << "/" << ann_path << ": pt5.y(" << box_new_3d.y5 << ") > pt1.y(" << box_new_3d.y1 << ")" << std::endl;
				}
				else if (box_new_3d.shape == 4 && box_new_3d.y6 > box_new_3d.y3) {
					LOG(INFO) << tar_path << "/" << ann_path << ": pt6.y(" << box_new_3d.y6 << ") > pt3.y(" << box_new_3d.y3 << ")" << std::endl;
				}
				else if (box_new_3d.shape == 4 && box_new_3d.y7 > box_new_3d.y2) {
					LOG(INFO) << tar_path << "/" << ann_path << ": pt7.y(" << box_new_3d.y7 << ") > pt2.y(" << box_new_3d.y2 << ")" << std::endl;
				}
				else if (box_new_3d.shape == 5 && box_new_3d.y5 > box_new_3d.y2) {
					LOG(INFO) << tar_path << "/" << ann_path << ": pt5.y(" << box_new_3d.y5 << ") > pt2.y(" << box_new_3d.y2 << ")" << std::endl;
				}
				else if (box_new_3d.shape == 5 && box_new_3d.y6 > box_new_3d.y4) {
					LOG(INFO) << tar_path << "/" << ann_path << ": pt6.y(" << box_new_3d.y6 << ") > pt4.y(" << box_new_3d.y4 << ")" << std::endl;
				}
				else if (box_new_3d.shape == 5 && box_new_3d.y7 > box_new_3d.y1) {
					LOG(INFO) << tar_path << "/" << ann_path << ": pt7.y(" << box_new_3d.y7 << ") > pt1.y(" << box_new_3d.y1 << ")" << std::endl;
				}
				else {
					return true;
				}
			}
			catch (std::exception const& e) {
			}
			return false;
		}

		void UDB::Open() {
			FILE* fp_r = NULL;
			FILE* fp_w = NULL;
			if (use_cache_) {
				fp_r = fopen(cache_file_, "rb");
				if (fp_r) {
					char ucv[32] = { NULL, };
					fread(ucv, strlen(UDB_CACHE_VER), 1, fp_r);
					if (strcmp(ucv, UDB_CACHE_VER) != 0) {
						fclose(fp_r);
						fp_r = NULL;
					}
				}
				if (fp_r == NULL && cache_write_) {
					fp_w = fopen(cache_temp_file_, "wb");
					char ucv[32] = { NULL, };
					strcpy(ucv, UDB_CACHE_VER);
					fwrite(ucv, strlen(UDB_CACHE_VER), 1, fp_w);
				}
			}

			tar_readers_.resize(db_files_.size());
			for (int i = 0; i < db_files_.size(); i++) {
				for (int j = 0; j < db_files_[i].first.size(); j++) {
					if (db_files_[i].first[j] == "") {
						CHECK(j != 0);
						tar_readers_[i].push_back(NULL);
					}
					else {
						if (tar_reader_pool_.find(db_files_[i].first[j]) == tar_reader_pool_.end()) {
							LOG(INFO) << "Loading... " << db_files_[i].first[j];
							tar_reader_pool_[db_files_[i].first[j]] = new TarReader(db_files_[i].first[j].c_str(), fp_r, fp_w);
						}
						tar_readers_[i].push_back(tar_reader_pool_[db_files_[i].first[j]]);
					}
				}
			}

			CHECK_EQ(clsidxlst_.size(), clslst_.size());
			for (int i = 0; i < clslst_.size(); i++)
				str2lower(clslst_[i]);
			for (int i = 0; i < clsidxlst_.size(); i++) {
				if (clsidxlst_[i] != -100000 && clsidxlst_[i] != -100001 && clsidxlst_[i] < 0) {
					LOG(INFO) << "class index " << clsidxlst_[i] << " is replaced to 0" << std::endl;
					clsidxlst_[i] = 0;
				}
			}

			for (int i = 0; i < clslst_.size(); i++) {
				clsmap_.insert(std::make_pair(clslst_[i], clsidxlst_[i]));
				if (clsinvmap_.find(clsidxlst_[i]) == clsinvmap_.end())
					clsinvmap_.insert(std::make_pair(clsidxlst_[i], std::vector<std::string>()));
				clsinvmap_[clsidxlst_[i]].push_back(clslst_[i]);
				LOG(INFO) << "Recognizable class: \"" << clslst_[i] << " => " << clsidxlst_[i] << "\"";
			}

			if (fp_r) {
				LOG(INFO) << "Loading... " << cache_file_;
				for (int taridx = 0, udb_points_index = 0; taridx < tar_readers_.size(); taridx++) {
					TarReader* tar_reader_img = tar_readers_[taridx][0];
					TarReader* tar_reader_ann = tar_readers_[taridx].size() >= 2 && tar_readers_[taridx][1] ? tar_readers_[taridx][1] : tar_readers_[taridx][0];
					TarReader* tar_reader_seg = tar_readers_[taridx].size() >= 3 && tar_readers_[taridx][2] ? tar_readers_[taridx][2] : tar_readers_[taridx][0];
					int udb_points_size;
					fread(&udb_points_size, sizeof(int), 1, fp_r);
					udb_points_.resize(udb_points_size, NULL);
					for (; udb_points_index < udb_points_size; udb_points_index++) {
						UDBPoint* cur_data = new UDBPoint(tar_reader_img, tar_reader_ann, tar_reader_seg);
						cur_data->load_from_fp(fp_r);
						udb_points_[udb_points_index] = cur_data;
						udb_points_map_[cur_data->key_] = cur_data;
					}
					for (auto it = udb_points_map_.begin(); it != udb_points_map_.end(); it++) {
						if (it->second->seq_len_ > 0) {
							CHECK(udb_points_map_.find(it->second->next_key_) != udb_points_map_.end());
							it->second->next_ptr_ = udb_points_map_[it->second->next_key_];
						}
					}
				}
				fread_map(&object_count1_, fp_r);
				fread_map(&object_count2_, fp_r);
				fread_map(&gt_count_, fp_r);
			}
			else {
				for (int taridx = 0, udb_points_index = 0; taridx < tar_readers_.size(); taridx++) {
					LOG(INFO) << "Parsing... " << tar_readers_[taridx][0]->path();
					TarReader* tar_reader_img = tar_readers_[taridx][0];
					TarReader* tar_reader_ann = tar_readers_[taridx].size() >= 2 && tar_readers_[taridx][1] ? tar_readers_[taridx][1] : tar_readers_[taridx][0];
					TarReader* tar_reader_seg = tar_readers_[taridx].size() >= 3 && tar_readers_[taridx][2] ? tar_readers_[taridx][2] : tar_readers_[taridx][0];
					std::vector<std::string> selected_entries;
					std::string selectedlst;

					// read the split file
					std::string datasplit_name = std::string("ImageSets/");
					datasplit_name += db_files_[taridx].second + std::string(".txt");
					if (tar_reader_ann->exists(datasplit_name)) {
						tar_reader_ann->read(datasplit_name, selectedlst);
					}
					else if (tar_reader_img->exists(datasplit_name)) {
						tar_reader_img->read(datasplit_name, selectedlst);
					}
					else if (tar_reader_seg->exists(datasplit_name)) {
						tar_reader_seg->read(datasplit_name, selectedlst);
					}
					else {
						LOG(ERROR)
							<< "Cannot find the data split file '"
							<< datasplit_name << "' in '" << db_files_[taridx].first[0] << "'";
						exit(-1);
					}

					std::vector<std::string> flist1, flist2;
					tar_reader_ann->listdir("Annotations", flist1);
					tar_reader_img->listdir("Annotations", flist2);

					if (flist1.size() == 0 && flist2.size() > 0) {
						tar_reader_ann = tar_reader_img;
					}

					// parse data split file
					stringstream lstss(selectedlst);
					std::string curline;
					for (; getline(lstss, curline);) {
						boost::trim(curline);
						if (curline.size() != 0)
							selected_entries.push_back(curline);
					}

					for (int i = 0; i < selected_entries.size(); i++) {
						std::string dataname = selected_entries[i];
						std::string annotation_path = "Annotations/" + dataname + ".xml";
						std::string jpg_path = "JPEGImages/" + dataname + ".jpg";
						std::string png_path = "JPEGImages/" + dataname + ".png";
						std::string bmp_path = "JPEGImages/" + dataname + ".bmp";
						std::string segmentation_path = "Segmentations/" + dataname + ".png";
						std::string ego_xy_path = "Ego_XY/" + dataname + ".xml";
						UDBPoint* cur_data = new UDBPoint(tar_reader_img, tar_reader_ann, tar_reader_seg, dataname);

						int box_2d_count = 0;
						int quad_2d_count = 0;
						int box_3d_count = 0;
						int box_new_3d_count = 0;

						if (use_img_) {
							if (tar_reader_img->exists(jpg_path))
								cur_data->img_path_ = jpg_path;
							else if (tar_reader_img->exists(png_path))
								cur_data->img_path_ = png_path;
							else if (tar_reader_img->exists(bmp_path))
								cur_data->img_path_ = bmp_path;
							else {
								LOG(ERROR) << "tar parsing error: cannot find a file " << jpg_path;
								exit(-1);
							}
						}

						if (use_lane_type_label_ || use_boundary_type_label_) {
							if (tar_reader_ann->exists(annotation_path)) {
								cur_data->ann_path_ = annotation_path;

								ptree pt;

								std::string data;
								tar_reader_ann->read(annotation_path, data);

								try {
									std::stringstream ss(data);
									read_xml(ss, pt);
								}
								catch (std::exception const& e) {
									LOG(ERROR) << "tar parsing error: " << annotation_path;
									exit(-1);
								}
								if (use_lane_type_label_) {
									try {
										int height = pt.get<int>("LaneBoundaryTypes.<xmlattr>.imageHeight");
										int width = pt.get<int>("LaneBoundaryTypes.<xmlattr>.imageWidth");
										cur_data->img_width_ = width;
										cur_data->img_height_ = height;
										BOOST_FOREACH(ptree::value_type const & v, pt.get_child("LaneBoundaryTypes")) {
											if (v.first == "LaneLines") {
												int lanelinenum = v.second.get<int>("<xmlattr>.LaneLineNum");
												cur_data->lane_type_label_.push_back(lanelinenum);
												BOOST_FOREACH(ptree::value_type const & vv, v.second) {
													if (vv.first == "LaneLine") {
														cur_data->lane_type_label_.push_back(vv.second.get<int>("<xmlattr>.id"));
														cur_data->lane_type_label_.push_back(vv.second.get<float>("<xmlattr>.typeShape"));
														cur_data->lane_type_label_.push_back(vv.second.get<float>("<xmlattr>.typeSD"));
														cur_data->lane_type_label_.push_back(vv.second.get<float>("<xmlattr>.typePos"));
														cur_data->lane_type_label_.push_back(vv.second.get<float>("<xmlattr>.typeColor"));
														cur_data->lane_type_label_.push_back(vv.second.get<float>("<xmlattr>.typeBicycle"));
													}
												}
											}
										}
									}
									catch (std::exception const& e) {
										LOG(ERROR) << "tar parsing error: cannot read lane type xml" << annotation_path;
										exit(-1);
									}
								}
								if (use_boundary_type_label_) {
									try {
										int height = pt.get<int>("LaneBoundaryTypes.<xmlattr>.imageHeight");
										int width = pt.get<int>("LaneBoundaryTypes.<xmlattr>.imageWidth");
										cur_data->img_width_ = width;
										cur_data->img_height_ = height;
										BOOST_FOREACH(ptree::value_type const & v, pt.get_child("LaneBoundaryTypes")) {
											if (v.first == "BoundaryLines") {
												int boundarylinenum = v.second.get<int>("<xmlattr>.BoundaryLineNum");
												cur_data->boundary_type_label_.push_back(boundarylinenum);
												BOOST_FOREACH(ptree::value_type const & vv, v.second) {
													if (vv.first == "BoundaryLine") {
														cur_data->boundary_type_label_.push_back(vv.second.get<int>("<xmlattr>.id"));
														cur_data->boundary_type_label_.push_back(vv.second.get<float>("<xmlattr>.typeShape"));
														cur_data->boundary_type_label_.push_back(vv.second.get<float>("<xmlattr>.typePos"));
													}
												}
											}
										}
									}
									catch (std::exception const& e) {
										LOG(ERROR) << "tar parsing error: cannot read boundary type xml" << annotation_path;
										exit(-1);
									}
								}
							}
						}

						if (use_ego_out_) {
							if (tar_reader_ann->exists(ego_xy_path)) {
								cur_data->ego_xy_path_ = ego_xy_path;
								/*<annotation>
								  <vpy>93< / vpy>
								  <npts>254< / npts>
								  <pts>
								  <x>-0.063021< / x>
								  <y>0.136458< / y>*/
								ptree pt;

								std::string data;
								tar_reader_ann->read(ego_xy_path, data);

								std::stringstream ss(data);
								read_xml(ss, pt);

								float xtmp = -1, ytmp = -1;
								BOOST_FOREACH(ptree::value_type const & v, pt.get_child("annotation").get_child("lpts")) {
									if (v.first == "x") {
										xtmp = std::stof(v.second.data());
									}
									if (v.first == "y") {
										ytmp = std::stof(v.second.data());
										cur_data->ego_xy_L_.push_back(cv::Point2f(xtmp, ytmp));
									}
								}

								BOOST_FOREACH(ptree::value_type const & v, pt.get_child("annotation").get_child("rpts")) {
									if (v.first == "x") {
										xtmp = std::stof(v.second.data());
									}
									if (v.first == "y") {
										ytmp = std::stof(v.second.data());
										cur_data->ego_xy_R_.push_back(cv::Point2f(xtmp, ytmp));
									}
								}


								cur_data->vp_x_ = 0.5;
								cur_data->vp_y_ = 0.5;

								try {
									cur_data->vp_x_ = pt.get_child("annotation").get<float>("vpx");;
								}
								catch (std::exception const& e) {
									LOG(ERROR) << "tar parsing error: cannot read vp x" << ego_xy_path;
									exit(-1);
								}

								try {
									cur_data->vp_y_ = pt.get_child("annotation").get<float>("vpy");
								}
								catch (std::exception const& e) {
									LOG(ERROR) << "tar parsing error: cannot read vp y" << ego_xy_path;
									exit(-1);
								}

							}
						}

						if (use_failsafe_) {
							if (tar_reader_ann->exists(annotation_path)) {
								cur_data->ann_path_ = annotation_path;

								ptree pt;

								std::string data;
								tar_reader_ann->read(annotation_path, data);

								try {
									std::stringstream ss(data);
									read_xml(ss, pt);
								}
								catch (std::exception const& e) {
									LOG(ERROR) << "tar parsing error: " << annotation_path;
									exit(-1);
								}

								int width = 0, height = 0;
								try {
									width = pt.get_child("annotation").get_child("size").get<int>("width");
								}
								catch (std::exception const& e) {
									LOG(ERROR) << "tar parsing error: cannot read image width " << annotation_path;
									exit(-1);
								}
								try {
									height = pt.get_child("annotation").get_child("size").get<int>("height");
								}
								catch (std::exception const& e) {
									LOG(ERROR) << "tar parsing error: cannot read image height " << annotation_path;
									exit(-1);
								}
								if (use_failsafe_) {
									try {
										cur_data->failsafe_ = pt.get_child("annotation").get<int>("failsafe");
									}
									catch (std::exception const& e) {
										LOG(ERROR) << "tar parsing error: cannot read failsafe" << annotation_path;
										exit(-1);
									}
								}
								cur_data->img_width_ = width;
								cur_data->img_height_ = height;

							}
						}

						if (use_od_ || use_3d_ || use_new_3d_ || use_od_ex_ || use_scene_ || use_tsr_cls_ || use_tlr_cls_ || use_tlr_blob_ || use_tlr_blobReg_ || use_meta_info_) {
							if (tar_reader_ann->exists(annotation_path)) {
								cur_data->ann_path_ = annotation_path;

								ptree pt;

								std::string data;
								tar_reader_ann->read(annotation_path, data);

								try {
									std::stringstream ss(data);
									read_xml(ss, pt);
								}
								catch (std::exception const& e) {
									LOG(ERROR) << "tar parsing error: " << annotation_path;
									exit(-1);
								}
								/*
								example OD, 3D, Attribute
								<annotation>
								  <size>
									<width>1920</width>
									<height>1080</height>
								  </size>
								  <object>
									<name>Pedestrian</name>
									<bndbox>
									  <xmin>220.45</xmin>
									  <ymin>144.35</ymin>
									  <xmax>289.12</xmax>
									  <ymax>301.68</ymax>
									</bndbox>
								  </object>
								  <object>
									<name>Car</name>
									<bndbox>
									  <xmin>1027.20</xmin>
									  <ymin>657.60</ymin>
									  <xmax>1155.72</xmax>
									  <ymax>770.31</ymax>
									</bndbox>
									<direction>8</direction> // 1~8
									<hexahedron>
									  <x1>1048.99</x1>
									  <y1>665.72</y1>
									  <x2>1153.93</x2>
									  <y2>770.66</y2>
									  <x3>1026.39</x3>
									  <y3>671.03</y3>
									  <x4>1126.02</x4>
									  <y4>752.24</y4>
									</hexahedron>
									<occluded>0</occluded> // 0, 0.25, 0.5, 0.75, 1
									<truncated>0</truncated> // 0, 0.25, 0.5, 0.75, 1
									<sit_stand>1</sit_stand> //0: sit, 1: stand
									<age>1</age> //-1: ignored, 1: adult, 2: child
									<gender>-1</gender> //-1: ignored, 1: male, 2: female
								  </object>
								</annotation>

								example new 3D:
								<File ObjectCount="2" ImageFile="Cont_20180315_181901_v1.06.00_R3_B0_00800.jpg", ImageWidth="1920", ImageHeight="1080">
									<Car3D Direction="7" Shape="1">
										<Point x="1548" y="326"/>
										<Point x="1668" y="326"/>
										<Point x="1548" y="519"/>
										<Point x="1668" y="519"/>
										<Point x="1508" y="346"/>
										<Point x="1508" y="499"/>
									</Car3D>
									<Car3D Direction="9" Shape="2">
										<Point x="986" y="431"/>
										<Point x="1145" y="431"/>
										<Point x="986" y="579"/>
										<Point x="1145" y="579"/>
										<Point x="1208" y="458"/>
										<Point x="1208" y="552"/>
									</Car3D>
								</File>
								*/
								int width = 0, height = 0;
								try {
									width = pt.get_child("annotation").get_child("size").get<int>("width");
								}
								catch (std::exception const& e) {
								}
								try {
									height = pt.get_child("annotation").get_child("size").get<int>("height");
								}
								catch (std::exception const& e) {
								}
								try {
									width = pt.get_child("File").get<int>("<xmlattr>.ImageWidth");
								}
								catch (std::exception const& e) {
								}
								try {
									height = pt.get_child("File").get<int>("<xmlattr>.ImageHeight");
								}
								catch (std::exception const& e) {
								}
								if (width == 0) {
									LOG(ERROR) << "tar parsing error: cannot read image width " << annotation_path;
									exit(-1);
								}
								if (height == 0) {
									LOG(ERROR) << "tar parsing error: cannot read image height " << annotation_path;
									exit(-1);
								}
								cur_data->img_width_ = width;
								cur_data->img_height_ = height;

								try {
									cur_data->next_key_ = pt.get_child("annotation").get<std::string>("next_key");
								}
								catch (std::exception const& e) {
								}
								try {
									cur_data->seq_len_ = pt.get_child("annotation").get<int>("seq_len");
								}
								catch (std::exception const& e) {
								}

								std::vector<bool> mandatory_class_find(mandatory_class_.size(), false);
								std::unordered_map<string, int> object_count1;
								std::unordered_map<string, int> object_count2;
								try {
									BOOST_FOREACH(ptree::value_type const & v, pt.get_child("annotation")) {
										if (v.first == "object") {
											ObjectGT object_gt;
											if (use_od_ || use_od_ex_) {
												std::string clsname;
												Box2D box_2d;
												Quad2D quad_2d;
												BoxAttribute box_attribute;
												if (use_od_quad_ && get_quad_data(v, width, height, tar_reader_ann->path(), annotation_path, clsname, box_2d, quad_2d, box_attribute)) {
													object_gt.box_2d_idx_ = cur_data->box_2d_.size();
													cur_data->box_2d_.push_back(box_2d);
													object_gt.quad_2d_idx_ = cur_data->quad_2d_.size();
													cur_data->quad_2d_.push_back(quad_2d);

													if (!box_attribute.empty()) {
														object_gt.box_attribute_idx_ = cur_data->box_attribute_.size();
														cur_data->box_attribute_.push_back(box_attribute);
													}
													if (box_2d.label > 0) {
														box_2d_count++;
														quad_2d_count++;
													}
													for (int k = 0; k < mandatory_class_.size(); k++) {
														if (box_2d.label == mandatory_class_[k]) {
															mandatory_class_find[k] = true;
															break;
														}
													}
													if (object_count1.find(clsname) == object_count1.end()) {
														object_count1[clsname] = 1;
													}
													else {
														object_count1[clsname]++;
													}
													string clsindex = class_index_to_string(box_2d.label);
													if (object_count2.find(clsindex) == object_count2.end()) {
														object_count2[clsindex] = 1;
													}
													else {
														object_count2[clsindex]++;
													}
												}
												else if (get_od_data(v, width, height, tar_reader_ann->path(), annotation_path, clsname, box_2d, box_attribute)) {
													object_gt.box_2d_idx_ = cur_data->box_2d_.size();
													cur_data->box_2d_.push_back(box_2d);
													if (!box_attribute.empty()) {
														object_gt.box_attribute_idx_ = cur_data->box_attribute_.size();
														cur_data->box_attribute_.push_back(box_attribute);
													}

													if (box_2d.label > 0)
														box_2d_count++;
													for (int k = 0; k < mandatory_class_.size(); k++) {
														if (box_2d.label == mandatory_class_[k]) {
															mandatory_class_find[k] = true;
															break;
														}
													}
													if (object_count1.find(clsname) == object_count1.end()) {
														object_count1[clsname] = 1;
													}
													else {
														object_count1[clsname]++;
													}
													string clsindex = class_index_to_string(box_2d.label);
													if (object_count2.find(clsindex) == object_count2.end()) {
														object_count2[clsindex] = 1;
													}
													else {
														object_count2[clsindex]++;
													}
												}
											}
											if (use_3d_ || use_od_ex_) {
												Box3D box_3d;
												if (get_3d_data(v, width, height, tar_reader_ann->path(), annotation_path, box_3d)) {
													object_gt.box_3d_idx_ = cur_data->box_3d_.size();
													cur_data->box_3d_.push_back(box_3d);
													box_3d_count++;
												}
											}
											if (!object_gt.empty()) {
												cur_data->object_gt_.push_back(object_gt);
											}
											if (use_tsr_cls_) {
												std::string clsname = get_name(v);
												cur_data->tsr_cls_ = from_clsname(clsname);
												if (cur_data->tsr_cls_ == INT_MAX) {
													LOG(ERROR) << "unrecognized class name " << clsname << " for " << cur_data->img_path_ << std::endl;
													exit(-1);
												}
											}
											if (use_tlr_cls_) {
												std::string clsname = get_name(v);
												cur_data->tlr_cls_ = from_clsname(clsname);
												if (cur_data->tlr_cls_ == INT_MAX) {
													LOG(ERROR) << "unrecognized class name " << clsname << " for " << cur_data->img_path_ << std::endl;
													exit(-1);
												}
											}
											if (use_tlr_blob_ || use_tlr_blobReg_) {
												BOOST_FOREACH(ptree::value_type const & tl, v.second) {
													if (tl.first == "tl_info") {
														cur_data->tlr_blobs_ = tl.second.get<int>("blobs") - 1;
														if (use_tlr_blobReg_) {
															cur_data->tlr_ct_pt_.push_back(cv::Point2i(tl.second.get<int>("ct_x0"), tl.second.get<int>("ct_y0")));
															cur_data->tlr_ct_pt_.push_back(cv::Point2i(tl.second.get<int>("ct_x1"), tl.second.get<int>("ct_y1")));
															cur_data->tlr_ct_pt_.push_back(cv::Point2i(tl.second.get<int>("ct_x2"), tl.second.get<int>("ct_y2")));
															cur_data->tlr_ct_pt_.push_back(cv::Point2i(tl.second.get<int>("ct_x3"), tl.second.get<int>("ct_y3")));
															cur_data->tlr_ct_pt_.push_back(cv::Point2i(tl.second.get<int>("ct_x4"), tl.second.get<int>("ct_y4")));
														}
													}
												}
											}
										}
										else if (use_scene_ && v.first == "scene") {
											try {
												cur_data->scene_lbl_.time = v.second.get<int>("time");
												cur_data->scene_lbl_.place = v.second.get<int>("place");
												cur_data->scene_lbl_.weather = v.second.get<int>("weather");
											}
											catch (std::exception const& e) {
												if (req_scene_) {
													LOG(ERROR) << "tar parsing error: cannot read scene label " << annotation_path;
													exit(-1);
												}
											}
										}
										else if (use_meta_info_ && v.first == "meta_info") {
											std::vector<int> weather_list;
											BOOST_FOREACH(ptree::value_type const & mt, v.second) {
												if (mt.first == "road") {
													std::string road = mt.second.data();
													if (road.compare("city") == 0)
														cur_data->meta_info_.road = 1;
													else if (road.compare("highway") == 0)
														cur_data->meta_info_.road = 2;
													else if (road.compare("rural") == 0)
														cur_data->meta_info_.road = 3;
													else if (road.compare("etc") == 0)
														cur_data->meta_info_.road = 4;
													else
														cur_data->meta_info_.road = -2;
												}
												if (mt.first == "timezone_item") {
													try {
														int timezone = std::stoi(mt.second.data());
														if (0 <= timezone && timezone <= 255)
															cur_data->meta_info_.timezone_item = timezone;
														else
															cur_data->meta_info_.timezone_item = -2;
													}
													catch (int exception) {
														cur_data->meta_info_.timezone_item = -2;
													}
												}
												if (mt.first == "weather_item") {
													std::string weather = mt.second.data();
													if (weather.compare("clean_road") == 0)
														weather_list.push_back(1);
													else if (weather.compare("wet_light_road") == 0)
														weather_list.push_back(2);
													else if (weather.compare("wet_medium_road") == 0)
														weather_list.push_back(3);
													else if (weather.compare("wet_severe_road") == 0)
														weather_list.push_back(4);
													else if (weather.compare("snow_light_road") == 0)
														weather_list.push_back(5);
													else if (weather.compare("snow_medium_road") == 0)
														weather_list.push_back(6);
													else if (weather.compare("snow_severe_road") == 0)
														weather_list.push_back(7);
													else if (weather.compare("light_reflection_light_road") == 0)
														weather_list.push_back(8);
													else if (weather.compare("light_reflection_medium_road") == 0)
														weather_list.push_back(9);
													else if (weather.compare("light_reflection_severe_road") == 0)
														weather_list.push_back(10);
													else if (weather.compare("road_etc") == 0)
														weather_list.push_back(11);
													else if (weather.compare("snow_light_sidewalk") == 0)
														weather_list.push_back(12);
													else if (weather.compare("snow_medium_sidewalk") == 0)
														weather_list.push_back(13);
													else if (weather.compare("snow_severe_sidewalk") == 0)
														weather_list.push_back(14);
													else if (weather.compare("fog_light") == 0)
														weather_list.push_back(15);
													else if (weather.compare("fog_medium") == 0)
														weather_list.push_back(16);
													else if (weather.compare("fog_severe") == 0)
														weather_list.push_back(17);
													else if (weather.compare("wiper_light") == 0)
														weather_list.push_back(18);
													else if (weather.compare("wiper_severe") == 0)
														weather_list.push_back(19);
													else
														weather_list.push_back(-2);
												}
											}
											for (int i = 0; i < weather_list.size(); i++) {
												if (i == 0)
													cur_data->meta_info_.weather_item_1 = weather_list[i];
												else if (i == 1)
													cur_data->meta_info_.weather_item_2 = weather_list[i];
												else if (i == 2)
													cur_data->meta_info_.weather_item_3 = weather_list[i];
												else if (i == 3)
													cur_data->meta_info_.weather_item_4 = weather_list[i];
											}
										}
									}
								}
								catch (std::exception const& e) {
								}

								try {
									BOOST_FOREACH(ptree::value_type const & v, pt.get_child("File")) {
										if (v.first == "Car3D") {
											ObjectGT object_gt;
											if (use_new_3d_ || use_od_ex_) {
												BoxNew3D box_new_3d;
												if (get_new_3d_data(v, width, height, tar_reader_ann->path(), annotation_path, box_new_3d)) {
													object_gt.box_new_3d_idx_ = cur_data->box_new_3d_.size();
													cur_data->box_new_3d_.push_back(box_new_3d);
													box_new_3d_count++;
												}
											}
											if (!object_gt.empty()) {
												cur_data->object_gt_.push_back(object_gt);
											}
										}
									}
								}
								catch (std::exception const& e) {
								}
								if (use_od_ || use_od_ex_) {
									for (int k = 0; k < mandatory_class_find.size(); k++) {
										if (!mandatory_class_find[k]) {
											box_2d_count = 0;
											break;
										}
									}
									if (box_2d_count > 0) {
										for (auto it : object_count1) {
											if (object_count1_.find(it.first) == object_count1_.end()) {
												object_count1_[it.first] = it.second;
											}
											else {
												object_count1_[it.first] += it.second;
											}
										}
										for (auto it : object_count2) {
											if (object_count2_.find(it.first) == object_count2_.end()) {
												object_count2_[it.first] = it.second;
											}
											else {
												object_count2_[it.first] += it.second;
											}
										}
									}
									if (gt_count_.find("od") == gt_count_.end()) {
										gt_count_["od"] = box_2d_count;
									}
									else {
										gt_count_["od"] += box_2d_count;
									}
									if (use_od_quad_) {
										if (gt_count_.find("quad") == gt_count_.end()) {
											gt_count_["quad"] = quad_2d_count;
										}
										else {
											gt_count_["quad"] += quad_2d_count;
										}
									}
								}
								if (use_3d_ || use_od_ex_) {
									if (gt_count_.find("3d") == gt_count_.end()) {
										gt_count_["3d"] = box_3d_count;
									}
									else {
										gt_count_["3d"] += box_3d_count;
									}
								}
								if (use_new_3d_ || use_od_ex_) {
									if (gt_count_.find("new_3d") == gt_count_.end()) {
										gt_count_["new_3d"] = box_new_3d_count;
									}
									else {
										gt_count_["new_3d"] += box_new_3d_count;
									}
								}
							}
							else {
								LOG(ERROR) << "tar parsing error: cannot find a file " << annotation_path;
								exit(-1);
							}
						}

						if (use_seg_) {
							if (tar_reader_seg->exists(segmentation_path))
								cur_data->seg_path_ = segmentation_path;
							else if (req_seg_) {
								LOG(ERROR) << "tar parsing error: cannot find a file " << segmentation_path;
								exit(-1);
							}
						}

						if (param_.use_blank_roi() ||
							((!req_od_ || box_2d_count > 0) &&
							(!req_3d_ || box_3d_count > 0) &&
								(!req_new_3d_ || box_new_3d_count > 0) &&
								(!req_od_ex_ || cur_data->object_gt_.size() > 0))) {
							udb_points_.push_back(cur_data);
							udb_points_map_[cur_data->key_] = cur_data;
						}
						else {
							delete cur_data;
						}
					}

					for (auto it = udb_points_map_.begin(); it != udb_points_map_.end(); it++) {
						if (it->second->seq_len_ > 0) {
							CHECK(udb_points_map_.find(it->second->next_key_) != udb_points_map_.end());
							it->second->next_ptr_ = udb_points_map_[it->second->next_key_];
						}
					}

					if (fp_w) {
						int udb_points_size = udb_points_.size();
						fwrite(&udb_points_size, sizeof(int), 1, fp_w);
						for (; udb_points_index < udb_points_size; udb_points_index++) {
							udb_points_[udb_points_index]->save_to_fp(fp_w);
						}
					}
				}
				if (fp_w) {
					fwrite_map(&object_count1_, fp_w);
					fwrite_map(&object_count2_, fp_w);
					fwrite_map(&gt_count_, fp_w);
					LOG(INFO) << "Successfully saved udb cache: " << cache_file_;
				}
			}

			if (fp_r)
				fclose(fp_r);
			if (fp_w) {
				fclose(fp_w);
				boost::filesystem::rename(cache_temp_file_, cache_file_);
			}

			CHECK_GT(udb_points_.size(), 0) << "No data was read";

			LOG(INFO) << "-----------------------------------------------------------------------";
			LOG(INFO) << "Total " << udb_points_.size() << " data were read from following files:";
			for (int i = 0; i < db_files_.size(); i++)
				LOG(INFO) << db_files_[i].first[0] << " -> " << db_files_[i].second;
			if (gt_count_.size() > 0) {
				LOG(INFO) << "-----------------------------------------------------------------------";
				for (auto it : gt_count_)
					LOG(INFO) << "Total number of gts of the \"" << it.first << "\": " << it.second;
			}
			if (object_count1_.size() > 0) {
				LOG(INFO) << "-----------------------------------------------------------------------";
				for (auto it : object_count1_)
					LOG(INFO) << "Total number of objects of the class \"" << it.first << "\": " << it.second;
				LOG(INFO) << "-----------------------------------------------------------------------";
				for (auto it : object_count2_)
					LOG(INFO) << "Total number of objects of the \"" << it.first << "\": " << it.second;
			}
			LOG(INFO) << "-----------------------------------------------------------------------";
		}

	}  // namespace db
}  // namespace caffe
#endif  // USE_OPENCV
