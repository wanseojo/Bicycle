#ifndef _ROAD_LANE_V3_H_
#define _ROAD_LANE_V3_H_
#include "regressor.h"
#include <vector>
//#include "tinyxml2.h"      //// �߰� ////

using namespace std;
//using namespace tinyxml2;  //// �߰� ////

//version 3.0

class LaneInfo
{
private:
	char string_buffer[256];
public:
	
	typedef enum{
		C1_NONE = 0, SOLID = 1, DASHED, CATS_EYE // 0 ����
	}CATEGORY1;
	typedef enum{
		C2_NONE = 0, SINGLE = 1, DOUBLE = 2, ACCESSORIE = 3
	}CATEGORY2;
	typedef enum{
		C3_NONE = 0, LEFT = 1, RIGHT = 2, UNCERTAIN
	}CATEGORY3;
	typedef enum{
		C4_NONE = 0, BRANCH, MERGED, UNABLE, OPPOSITE_SIDE
	}CATEGORY4;
	typedef enum {
		C5_NONE = 0, WHITE, YELLOW, BLUE, ETC
	}CATEGORY5; 
	typedef enum {
		C6_NONE = 0, BICYCLE
	}CATEGORY6;
	//////////////////////// �߰� ////////////////////////

	int type1 = 0; // Solid, Dashed, Bot_dot
	int type2 = 0; // Single, Double line	
	int type3 = 0; // Left, Right, Uncertain
	int type4 = 0; // Branch, Merged, Unable, Opposite_side
	int type5 = 0; // WHITE, YELLOW, ORANGE, BLUE, ETC 
	int type6 = 0; // BICYCLE 
	//////////////////////// �߰� ////////////////////////

	std::vector<std::pair<float, float>> occlusions_top_bottom_;

	void Reset(){
		type1 = 0;// Solid, Dashed, Bot_dot
		type2 = 0; //Single, Double line	
		type3 = 0; //Left, Right, Uncertain
		type4 = 0; //Branch, Merged, Unable, Boundary, Curb Top, Curb Side
		type5 = 0; // WHITE, YELLOW, ORANGE, BLUE, ETC 
		type6 = 0; // BICYCLE 
		//////////////////////// �߰� ////////////////////////
		occlusions_top_bottom_.clear();
	}

	bool isEmpty() { return type1 == 0 && type2 == 0 && type3 == 0; }
	int GetType1() { return type1; }
	int GetType2() { return type2; }
	int GetType3() { return char(type3); }
	int GetType3_ID() { return type3>>8; }
	int GetType4() { return type4; }
	int GetType5() { return char(type5); }
	int GetType6() { return char(type6); }
	//////////////////////// �߰� ////////////////////////

	void SetType1(CATEGORY1 type) { type1 = type; }
	void SetType2(CATEGORY2 type) { type2 = type; }
	void SetType3(CATEGORY3 type, int id = 0) { type3 = char(type) + (id<<8); }
	void SetType4(CATEGORY4 type) { type4 = type; }
	void SetType5(CATEGORY5 type) { type5 = type; }
	void SetType6(CATEGORY6 type) { type6 = type; }
	//////////////////////// �߰� ////////////////////////

	const char *GetInfoText() {
		if (isEmpty()) {
			sprintf_s(string_buffer, "NONE");
			return string_buffer;
		}

		int type1 = GetType1();
		switch (type1)
		{
		case SOLID:
			sprintf_s(string_buffer, "��_");
			break;
		case DASHED:
			sprintf_s(string_buffer, "��_");
			break;
		case CATS_EYE:
			sprintf_s(string_buffer, "Ĺ_");
			break;
		default:
			sprintf_s(string_buffer, "");
			break;
		}
		int type2 = GetType2();
		switch (type2)
		{
		case SINGLE:
			strcat_s(string_buffer, "��_");
			break;
		case DOUBLE:
			strcat_s(string_buffer, "��_");
			break;
		case ACCESSORIE:
			strcat_s(string_buffer, "��_");
			break;
		default:
			strcat_s(string_buffer, "");
			break;
		}

		char tmp[256];
		int type3 = GetType3();
		switch (type3)
		{
		case C3_NONE:
			strcat_s(string_buffer, "��");
			break;
		case LEFT:
			sprintf_s(tmp, "��%d", GetType3_ID() + 1);
			strcat_s(string_buffer, tmp);
			break;
		case RIGHT:
			sprintf_s(tmp, "��%d", GetType3_ID() + 1);
			strcat_s(string_buffer, tmp);
			break;
		case UNCERTAIN:
			strcat_s(string_buffer, "��");
			break;
		default:
			break;
		}

		switch (GetType4())
		{
		case C4_NONE:
			strcat_s(string_buffer, "");
			break;
		case MERGED:
			strcat_s(string_buffer, "_��");
			break;
		case BRANCH:
			strcat_s(string_buffer, "_��");
			break;
		case UNABLE:
			strcat_s(string_buffer, "_��");
			break;
		case OPPOSITE_SIDE:
			strcat_s(string_buffer, "_��");
			break;
		}
		//////////////////////// �߰� ////////////////////////
		int type5 = GetType5();
		switch (type5)
		{
		case C5_NONE:
			strcat_s(string_buffer, "_NoColor");
			break;
		case WHITE:
			strcat_s(string_buffer, "_��");
			break;
		case YELLOW:
			strcat_s(string_buffer, "_��");
			break;
		case BLUE:
			strcat_s(string_buffer, "_��");
			break;
		case ETC:
			strcat_s(string_buffer, "_��Ÿ");
			break;
		default:
			break;
		}

		int type6 = GetType6();
		switch (type6)
		{
		case C6_NONE:
			break;
		case BICYCLE:
			strcat_s(string_buffer, "_������");
			break;
		default:
			break;
		}
		//////////////////////////////////////////////////////

		return string_buffer;
	};
};

class BoundaryInfo
{
private:
	char string_buffer[256];
public:

	typedef enum {
		C3_NONE = 0, LEFT = 1, RIGHT = 2/*, UNCERTAIN*/
	}CATEGORY3;
	typedef enum {
		BOUNDARY_NONE = 0, WALLS, STATIONARY_VEHICLES, GUARDRAIL, STATIC_LANE_SEPARATOR, CURBS, LANE_SEPARATOR, 
		PLASTIC_WALL, DRUM, BEACONS, CONE, ROAD_EDGE, INNER_PARKINGSPACE, UNEXEPLAINABLE, BOUNDARY_STRUCTURE_ETCS, BOUNDARY_ETCS
	}BOUNDARY;

	int type3 = 0; // Left, Right, Uncertain
	int BoundaryType = 0; // WALLS, STATIONARY_VEHICLES, GUARDRAIL, CURBS, LANE_SEPARATOR, BEACONS, ROAD_EDGE, BOUNDARY_ETCS
	int Situation = 0;
	std::vector<std::pair<float, float>> occlusions_top_bottom_;

	void Reset() {
		type3 = 0; //Left, Right, Uncertain
		BoundaryType = 0;
		Situation = 0;
		occlusions_top_bottom_.clear();
	}

	bool isEmpty() { return type3 == 0 && BoundaryType == 0; }
	int GetType3() { return char(type3); }
	int GetType3_ID() { return type3 >> 8; }
	int GetBoundaryType() { return BoundaryType; }

	void SetType3(CATEGORY3 type, int id = 0) { type3 = char(type) + (id << 8); }
	void SetBoundaryType(BOUNDARY type) { BoundaryType = type; }

	const char *GetInfoText() {
		if (isEmpty()) {
			sprintf_s(string_buffer, "NONE");
			return string_buffer;
		}
		int BoundaryType = GetBoundaryType();

		char tmp[256];
		int type3 = GetType3();
		int type3_id = GetType3_ID() + 1;
		switch (type3)
		{
		case C3_NONE:
			strcat_s(string_buffer, "��");
			break;
		case LEFT:
			if (type3_id >= 5) {
				sprintf_s(string_buffer, "��_��");
				break;
			}
			sprintf_s(string_buffer, "��%d", GetType3_ID() + 1);
			break;
		case RIGHT:
			if (type3_id >= 5) {
				sprintf_s(string_buffer, "��_��");
				break;
			}
			sprintf_s(string_buffer, "��%d", GetType3_ID() + 1);
			break;
		/*case UNCERTAIN:
			sprintf_s(string_buffer, "��");
			break;*/
		default:
			break;
		}

		switch (BoundaryType) {
		case BOUNDARY_NONE:
			strcat_s(string_buffer, "");
			break;
		case WALLS:
			strcat_s(string_buffer, "_����");
			break;
		case STATIONARY_VEHICLES:
			strcat_s(string_buffer, "_��������");
			break;
		case GUARDRAIL:
			strcat_s(string_buffer, "_���巹��");
			break;
		case STATIC_LANE_SEPARATOR:
			strcat_s(string_buffer, "_�����и���");
			break;
		case CURBS:
			strcat_s(string_buffer, "_����");
			break;
		case LANE_SEPARATOR:
			strcat_s(string_buffer, "_�ӽúи���");
			break;
		case PLASTIC_WALL:
			strcat_s(string_buffer, "_�ö�ƽ_��");
			break;
		case DRUM:
			strcat_s(string_buffer, "_�巳");
			break;
		case BEACONS:
			strcat_s(string_buffer, "_����");
			break;
		case CONE:
			strcat_s(string_buffer, "_��");
			break;
		case ROAD_EDGE:
			strcat_s(string_buffer, "_�����ڸ�");
			break;
		case INNER_PARKINGSPACE:
			strcat_s(string_buffer, "_�ǳ�����");
			break;
		case UNEXEPLAINABLE:
			strcat_s(string_buffer, "_����Ұ�");
			break;
		case BOUNDARY_STRUCTURE_ETCS:
			strcat_s(string_buffer, "_������");
			break;
		case BOUNDARY_ETCS:
			strcat_s(string_buffer, "_��Ÿ");
			break;
		default:
			strcat_s(string_buffer, "_��Ÿ");
			break;
		}
		return string_buffer;
	};
};

class RoadMarkerInfo
{
private:
	char string_buffer[256];
public:

	typedef enum {
		ROAD_MARKER_TYPE_HARD_NEGATIVE = 0,
		ROAD_MARKER_TYPE_STOP_LINE = 1,
		ROAD_MARKER_TYPE_CROSSWALK = 2,
		ROAD_MARKER_TYPE_ARROW = 3,
		ROAD_MARKER_TYPE_SPEED_BUMP = 4
	}RoadMakerType;	

	RoadMakerType type; // Solid, Dashed, Bot_dot
	void Reset() {
		type = ROAD_MARKER_TYPE_HARD_NEGATIVE;
	}

	RoadMakerType GetType()const {
		return type;
	}
	const char *GetInfoText() {
		RoadMakerType type = GetType();
		switch (type)
		{
		case ROAD_MARKER_TYPE_HARD_NEGATIVE:
			sprintf_s(string_buffer, "�����");
			break;
		case ROAD_MARKER_TYPE_STOP_LINE:
			sprintf_s(string_buffer, "������");
			break;
		case ROAD_MARKER_TYPE_CROSSWALK:
			sprintf_s(string_buffer, "Ⱦ�ܺ���");
			break;
		case ROAD_MARKER_TYPE_ARROW:
			sprintf_s(string_buffer, "��������");
			break;
		case ROAD_MARKER_TYPE_SPEED_BUMP:
			sprintf_s(string_buffer, "������");
			break;
		default:
			sprintf_s(string_buffer, "");
			break;
		}
		return string_buffer;
	};
};

class RoadMarkingPolygon {
public:
	RoadMarkingPolygon() {
		road_maker_type_.Reset();
	}
	~RoadMarkingPolygon() {}

	int GetPolygonPointNum() const { return polygon_.size(); }
	PPOINTF GetPoint(int idx) const { return polygon_[idx]; }
	void SetRoadMarkType(const RoadMarkerInfo::RoadMakerType type) {
		road_maker_type_.type = type;
	}
	RoadMarkerInfo GetRoadMarkerInfo() const { return road_maker_type_; }
	vector<PPOINTF> GetPolygon() const { 
		return polygon_; 
	}
	vector<PPOINTF> GetPoints() const { return polygon_; }
	bool PtInPolygon(PPOINTF point) {
		int i, j, nvert = polygon_.size();
		bool c = false;

		for (i = 0, j = nvert - 1; i < nvert; j = i++) {
			if (((polygon_[i].y >= point.y) != (polygon_[j].y >= point.y)) &&
				(point.x <= (polygon_[j].x - polygon_[i].x) * (point.y - polygon_[i].y) / (polygon_[j].y - polygon_[i].y) + polygon_[i].x)
				)
				c = !c;
		}
		return c;
	}
	bool FindPointIndexPtInPolygonPoint(PPOINTF point) {
		int i, j, nvert = polygon_.size();
		bool c = false;

		for (i = 0, j = nvert - 1; i < nvert; j = i++) {
			if (((polygon_[i].y >= point.y) != (polygon_[j].y >= point.y)) &&
				(point.x <= (polygon_[j].x - polygon_[i].x) * (point.y - polygon_[i].y) / (polygon_[j].y - polygon_[i].y) + polygon_[i].x)
				)
				c = !c;
		}
		return c;
	}
	bool SetPoints(const vector<PPOINTF> &points) {
		if (points.size() < 3) return false;
		polygon_ = points;
	}
protected:
	RoadMarkerInfo road_maker_type_;
	std::vector<PPOINTF> polygon_;
private:
	
};

class WorkingPolygon : public RoadMarkingPolygon
{
public:
	void PushBackPoint(PPOINTF point) {
		polygon_.push_back(point); 
	}
	void PopBackPoint() {
		polygon_.pop_back();
	}
	void RemovePoint(int idx) { 
		assert(idx >= 0);
		assert(idx < polygon_.size());
		polygon_.erase(polygon_.begin() + idx); 
	}

	void Reset() { polygon_.clear();}
	
	double ix(int idx) const { return polygon_[idx].x; }
	double iy(int idx) const { return polygon_[idx].y; }
	double vx(int idx) const { return mapi2v_x(ix(idx)); }
	double vy(int idx) const { return mapi2v_y(iy(idx)); }

	PPOINTF ixy(int idx) const {return polygon_[idx];}
	PPOINTF vxy(int idx) const { return PPOINTF(mapi2v(ixy(idx))); }	
protected:

private:

};

class LaneLine
{
public:
	LaneLine(){
		top_y_ = 0; bottom_y_ = 0;
	}
	~LaneLine(){}
	vector<double> spline_x_, spline_y_;
	vector<double> line_r_;

	double top_y_, bottom_y_;
	svld::tk::spline spline_xy_model_;
	svld::tk::spline spline_ry_model_;

	LaneInfo info;
	
	static bool comp(PPOINT3F &a, PPOINT3F &b) {
		return (a.y < b.y);
	}
	bool GenerateModels(){
		std::vector<PPOINT3F> xyz;
		if(spline_x_.size() < 3){
			top_y_ = 0; bottom_y_ = 0;
			return false;
		}
		for(int i = 0; i < spline_x_.size(); ++i) xyz.push_back(PPOINT3F(spline_x_[i], spline_y_[i], line_r_[i]));
		sort(xyz.begin(), xyz.end(), comp);
		for(int i = xyz.size() - 1; i--;){ if(xyz[i].y == xyz[i + 1].y){ xyz.erase(xyz.begin() + i + 1); } }
		if(xyz.size() < 3){
			top_y_ = 0; bottom_y_ = 0;
			return false;
		}
		top_y_ = xyz[0].y;
		bottom_y_ = xyz[xyz.size() - 1].y;

		std::vector<double> x, y, r;
		for(int i = 0; i < xyz.size(); i++){
			x.push_back(xyz[i].x); y.push_back(xyz[i].y);  r.push_back(xyz[i].r);
		}
		spline_xy_model_.set_points(y, x);
		spline_ry_model_.set_points(y, r, false);
		//line_r_model_.LeastSquareFit(&spline_y_[0], &line_r_[0], spline_y_.size());
		return true;
	}
	PPOINT3F EstimatePoint(double y){
		return PPOINT3F(spline_xy_model_(y), y, spline_ry_model_(y));
	}
protected:
	
};

class WorkingLaneLine : public LaneLine
{
public:
	WorkingLaneLine(){}
	~WorkingLaneLine(){}

	bool initialized() { return initialized_; }
	int element_size() const { return spline_x_.size(); }
	void erase_last_one() { erase_element(element_size() - 1); }

	double ix(int idx) const { return spline_x_[idx]; }
	double iy(int idx) const { return spline_y_[idx]; }
	double ir(int idx) const { return line_r_[idx]; }
	double vx(int idx) const { return mapi2v_x(ix(idx)); }
	double vy(int idx) const { return mapi2v_y(iy(idx)); }
	double vr(int idx) const { return mapi2v_r(ir(idx)); }

	PPOINT3F ixyr(int idx) const {
		return PPOINT3F(spline_x_[idx], spline_y_[idx], line_r_[idx]);
	}
	void set_ixyr(PPOINT3F &pt3, int idx) {
		spline_x_[idx] = pt3.x;
		spline_y_[idx] = pt3.y;
		line_r_[idx] = pt3.r;
		GenerateModels();
	}
	PPOINT3F vxyr(int idx) const {
		return PPOINT3F(mapi2v(ixyr(idx)));
	}

	PPOINT3F reg_ixyr(double y) const {
		if(initialized_)	return PPOINT3F(spline_xy_model_(y), y, spline_ry_model_(y));
		return PPOINT3F(FLT_MAX, FLT_MAX, FLT_MAX);
	}
	PPOINT3F reg_vxyr(double y) const {
		if(initialized_)	return mapi2v(PPOINT3F(spline_xy_model_(y), y, spline_ry_model_(y)));
		return PPOINT3F(FLT_MAX, FLT_MAX, FLT_MAX);
	}
	double reg_ix(double y) const {
		if(initialized_) return spline_xy_model_(y);
		return FLT_MAX;
	}
	double reg_ir(double y) const {
		if(initialized_) return spline_ry_model_(y);
		return line_r_.size() ? line_r_[line_r_.size()-1] : 10;
	}
	double reg_vx(double y) const {
		if(initialized_) mapi2v_x(reg_ix(y));
		return FLT_MAX;
	}
	double reg_vr(double y) const {
		return mapi2v_r(reg_ir(y));
	}

	void erase_element(int idx){
		if(idx < 0 || idx >= element_size()) return;
		spline_x_.erase(spline_x_.begin() + idx);
		spline_y_.erase(spline_y_.begin() + idx);
		line_r_.erase(line_r_.begin() + idx);
		GenerateModels();
	}

	void Reset(){
		spline_x_.clear();
		spline_y_.clear();
		line_r_.clear();
		info.Reset();
		GenerateModels();
	}

	bool GenerateModels(){
		if(LaneLine::GenerateModels()){
			initialized_ = true;
		} else {
			initialized_ = false;
		}
		return initialized_;
	}

	void AddPoint3(const PPOINT3F &point3){
		spline_x_.push_back(point3.x);
		spline_y_.push_back(point3.y);
		line_r_.push_back(point3.r);
	}

protected:

private:
	bool initialized_ = false;
};

class BoundaryLine
{
public:
	BoundaryLine() {
		top_y_ = 0; bottom_y_ = 0;
	}
	~BoundaryLine() {}
	vector<double> spline_x_, spline_y_;
	vector<double> line_r_;

	double top_y_, bottom_y_;
	svld::tk::spline spline_xy_model_;
	svld::tk::spline spline_ry_model_;

	BoundaryInfo info;

	static bool comp(PPOINT3F &a, PPOINT3F &b) {
		return (a.y < b.y);
	}
	bool GenerateModels() {
		std::vector<PPOINT3F> xyz;
		if (spline_x_.size() < 3) {
			top_y_ = 0; bottom_y_ = 0;
			return false;
		}
		for (int i = 0; i < spline_x_.size(); ++i) xyz.push_back(PPOINT3F(spline_x_[i], spline_y_[i], line_r_[i]));
		sort(xyz.begin(), xyz.end(), comp);
		for (int i = xyz.size() - 1; i--;) { if (xyz[i].y == xyz[i + 1].y) { xyz.erase(xyz.begin() + i + 1); } }
		if (xyz.size() < 3) {
			top_y_ = 0; bottom_y_ = 0;
			return false;
		}
		top_y_ = xyz[0].y;
		bottom_y_ = xyz[xyz.size() - 1].y;

		std::vector<double> x, y, r;
		for (int i = 0; i < xyz.size(); i++) {
			x.push_back(xyz[i].x); y.push_back(xyz[i].y);  r.push_back(xyz[i].r);
		}
		spline_xy_model_.set_points(y, x);
		spline_ry_model_.set_points(y, r, false);
		//line_r_model_.LeastSquareFit(&spline_y_[0], &line_r_[0], spline_y_.size());
		return true;
	}
	PPOINT3F EstimatePoint(double y) {
		return PPOINT3F(spline_xy_model_(y), y, spline_ry_model_(y));
	}
protected:

};

class WorkingBoundaryLine : public BoundaryLine
{
public:
	WorkingBoundaryLine() {}
	~WorkingBoundaryLine() {}

	bool initialized() { return initialized_; }
	int element_size() const { return spline_x_.size(); }
	void erase_last_one() { erase_element(element_size() - 1); }

	double ix(int idx) const { return spline_x_[idx]; }
	double iy(int idx) const { return spline_y_[idx]; }
	double ir(int idx) const { return line_r_[idx]; }
	double vx(int idx) const { return mapi2v_x(ix(idx)); }
	double vy(int idx) const { return mapi2v_y(iy(idx)); }
	double vr(int idx) const { return mapi2v_r(ir(idx)); }

	PPOINT3F ixyr(int idx) const {
		return PPOINT3F(spline_x_[idx], spline_y_[idx], line_r_[idx]);
	}
	void set_ixyr(PPOINT3F &pt3, int idx) {
		spline_x_[idx] = pt3.x;
		spline_y_[idx] = pt3.y;
		line_r_[idx] = pt3.r;
		GenerateModels();
	}
	PPOINT3F vxyr(int idx) const {
		return PPOINT3F(mapi2v(ixyr(idx)));
	}

	PPOINT3F reg_ixyr(double y) const {
		if (initialized_)	return PPOINT3F(spline_xy_model_(y), y, spline_ry_model_(y));
		return PPOINT3F(FLT_MAX, FLT_MAX, FLT_MAX);
	}
	PPOINT3F reg_vxyr(double y) const {
		if (initialized_)	return mapi2v(PPOINT3F(spline_xy_model_(y), y, spline_ry_model_(y)));
		return PPOINT3F(FLT_MAX, FLT_MAX, FLT_MAX);
	}
	double reg_ix(double y) const {
		if (initialized_) return spline_xy_model_(y);
		return FLT_MAX;
	}
	double reg_ir(double y) const {
		if (initialized_) return spline_ry_model_(y);
		return line_r_.size() ? line_r_[line_r_.size() - 1] : 10;
	}
	double reg_vx(double y) const {
		if (initialized_) mapi2v_x(reg_ix(y));
		return FLT_MAX;
	}
	double reg_vr(double y) const {
		return mapi2v_r(reg_ir(y));
	}

	void erase_element(int idx) {
		if (idx < 0 || idx >= element_size()) return;
		spline_x_.erase(spline_x_.begin() + idx);
		spline_y_.erase(spline_y_.begin() + idx);
		line_r_.erase(line_r_.begin() + idx);
		GenerateModels();
	}

	void Reset() {
		spline_x_.clear();
		spline_y_.clear();
		line_r_.clear();
		info.Reset();
		GenerateModels();
	}

	bool GenerateModels() {
		if (BoundaryLine::GenerateModels()) {
			initialized_ = true;
		}
		else {
			initialized_ = false;
		}
		return initialized_;
	}

	void AddPoint3(const PPOINT3F &point3) {
		spline_x_.push_back(point3.x);
		spline_y_.push_back(point3.y);
		line_r_.push_back(point3.r);
	}

protected:

private:
	bool initialized_ = false;
};

struct CurrentPoint{
	PPOINT3F point;
	bool regression_r = true;
};

class RoadLaneManager
{
public:
	RoadLaneManager();
	RoadLaneManager(const char *szpath);
	~RoadLaneManager();

	volatile bool IsZF() { return ((float)image_height_ / image_width_ > 0.7); }

	bool ReadFile(const char *szpath);
	bool WriteFile(const char *szpath);
	void Reset(int image_width, int image_height);

	void AddLine(const LaneLine &line);
	void ResetLine(const LaneLine &line, int idx);
	void RemoveLine(int idx);
	void AddPolygon(const RoadMarkingPolygon &polygon);
	void ResetPoygon(const RoadMarkingPolygon &polygon, int idx);
	void RemovePoygon(int idx);
	void AddBoundary(const BoundaryLine &line);
	void ResetBoundary(const BoundaryLine &line, int idx);
	void RemoveBoundary(int idx);
	void SetVPYRatio(double vp_y_ratio);
	void SetVPXRatio(double vp_x_ratio);
	void SetImageSize(int width, int height);
	void SetToolVersion(string str_version);

	int GetSizeLaneLine() { return lines_.size(); }
	int GetSizeRoadMarking() { return polygons_.size(); }
	int GetSizeBoundary() { return boundarys_.size(); }

	int GetImageW() { return image_width_; }
	int GetImageH() { return image_height_; }

	LaneLine *line_ptr(int idx) {
		if(idx >= 0 && idx < lines_.size())
			return &lines_[idx];
		else return NULL;
	}

	LaneLine line(int idx) {
		if(idx >= 0 && idx < lines_.size())
			return lines_[idx];
		else return LaneLine();
	}

	RoadMarkingPolygon *roadmarking_ptr(int idx) {
		if (idx >= 0 && idx < polygons_.size())
			return &polygons_[idx];
		else return NULL;
	}

	RoadMarkingPolygon roadmarking(int idx) {
		if (idx >= 0 && idx < polygons_.size())
			return polygons_[idx];
		else return RoadMarkingPolygon();
	}

	BoundaryLine *boundary_ptr(int idx) {
		if (idx >= 0 && idx < boundarys_.size())
			return &boundarys_[idx];
		else return NULL;
	}

	BoundaryLine boundary(int idx) {
		if (idx >= 0 && idx < boundarys_.size())
			return boundarys_[idx];
		else return BoundaryLine();
	}

	void SortingLength();

	double vp_x_ratio() { return vp_x_ratio_; }
	double vp_y_ratio() { return vp_y_ratio_; }
	bool has_vp() { return has_vp_; }
	void set_has_vp(bool vp) { has_vp_ = vp; }

	//tinyxml2::XMLDocument xmlDoc;
private:
	vector<RoadMarkingPolygon> polygons_;
	vector<LaneLine> lines_;	
	vector<BoundaryLine> boundarys_;
	double vp_y_ratio_;
	double vp_x_ratio_;
	bool has_vp_;
	int image_width_;
	int image_height_;
	string str_tool_version_;
protected:

};

struct BoundaryTypes {
	int id;
	int typePos;
	int typeShape;
};

struct LineTypes {
	int id;
	int typeShape;
	int typeSD;
	int typePos;
	int typeColor;
	int typeBicycle;
};
#endif