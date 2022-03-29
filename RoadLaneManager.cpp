#include "stdafx.h"
#include "RoadLaneManager.h"

#define DEFUALT_VP_Y 0.5
#define DEFUALT_VP_X 0.5
#define DEFUALT_BOUNDARY_W 5

RoadLaneManager::RoadLaneManager()
{
	image_width_ = 0;
	image_height_ = 0;
	vp_y_ratio_ = DEFUALT_VP_Y;
	vp_x_ratio_ = DEFUALT_VP_X;
	has_vp_ = false;
}
RoadLaneManager::RoadLaneManager(const char *szpath)
{
	image_width_ = 0;
	image_height_ = 0;
	vp_y_ratio_ = DEFUALT_VP_Y;
	vp_x_ratio_ = DEFUALT_VP_X;
	has_vp_ = false;
	ReadFile(szpath);
}
RoadLaneManager::~RoadLaneManager()
{
	Reset(0,0);
}

bool RoadLaneManager::ReadFile(const char *szpath)
{
	tinyxml2::XMLDocument xmlDoc;
	//파일 연결
	XMLError eResult = xmlDoc.LoadFile(szpath);

	if (eResult != XMLError::XML_SUCCESS)
	{
		printf("파일 로드 실패  - \"%s\".\n", szpath);
		return false;
	}
	
	XMLNode * pRoot = xmlDoc.FirstChild();
	if (pRoot == nullptr) 
		return XML_ERROR_FILE_READ_ERROR;
	
	XMLElement *pElement = xmlDoc.RootElement();
	const char *sztmp;
	eResult = pElement->QueryStringAttribute("toolVersion", &sztmp);
	//str_tool_version_ = sztmp;
	eResult = pElement->QueryIntAttribute("imageWidth", &image_width_);
	eResult = pElement->QueryIntAttribute("imageHeight", &image_height_);

	pElement = pRoot->FirstChildElement("VP");
	if (pElement == nullptr) 
		return XML_ERROR_PARSING_ELEMENT;
	eResult = pElement->QueryBoolAttribute("hasVP", &has_vp_);
	eResult = pElement->QueryDoubleAttribute("y_ratio", &vp_y_ratio_);
	eResult = pElement->QueryDoubleAttribute("x_ratio", &vp_x_ratio_);

	pElement = pRoot->FirstChildElement("Splines");
	int Spline_num = 0, Spline_count = 0;
	eResult = pElement->QueryIntAttribute("splineNum", &Spline_num);

	XMLElement * pSplineElement = pElement->FirstChildElement("Spline");
	lines_.resize(Spline_num);
	for (int i = 0; i < Spline_num; i++) {
		if (pSplineElement == nullptr) 
			return XML_ERROR_PARSING_ELEMENT;
		
		int point_num;
		eResult = pSplineElement->QueryIntAttribute("type1", &lines_[i].info.type1);
		eResult = pSplineElement->QueryIntAttribute("type2", &lines_[i].info.type2);
		eResult = pSplineElement->QueryIntAttribute("type3", &lines_[i].info.type3);
		eResult = pSplineElement->QueryIntAttribute("type4", &lines_[i].info.type4);
		eResult = pSplineElement->QueryIntAttribute("type5", &lines_[i].info.type5);
		eResult = pSplineElement->QueryIntAttribute("type6", &lines_[i].info.type6);
		eResult = pSplineElement->QueryIntAttribute("pointNum", &point_num);
		eResult = pSplineElement->QueryIntAttribute("occNum", &Spline_count);

		XMLElement * pPointElement = pSplineElement->FirstChildElement("Point");
		lines_[i].spline_x_.resize(point_num);
		lines_[i].spline_y_.resize(point_num);
		lines_[i].line_r_.resize(point_num);
		for (int j = 0; j < point_num; j++) {	
			eResult = pPointElement->QueryDoubleAttribute("x", &lines_[i].spline_x_[j]);
			eResult = pPointElement->QueryDoubleAttribute("y", &lines_[i].spline_y_[j]);
			eResult = pPointElement->QueryDoubleAttribute("r", &lines_[i].line_r_[j]);	
			pPointElement = pPointElement->NextSiblingElement("Point");
		}
				
		lines_[i].info.occlusions_top_bottom_.resize(Spline_count);
		XMLElement * pOccElement = pSplineElement->FirstChildElement("Occlusion");   // 가려짐
		for(int k = 0; k < Spline_count; k++){
			float top_y, bottom_y;

			eResult = pOccElement->QueryFloatAttribute("top", &top_y);
			eResult = pOccElement->QueryFloatAttribute("bottom", &bottom_y);
			
			lines_[i].info.occlusions_top_bottom_[k] = std::make_pair(top_y, bottom_y);
			pOccElement = pOccElement->NextSiblingElement("Occlusion");
		}

		lines_[i].GenerateModels();
		pSplineElement = pSplineElement->NextSiblingElement("Spline");
	}

	int polygon_num = 0;
	pElement = pRoot->FirstChildElement("Polygons");
	if (pElement == nullptr)
		return XML_ERROR_PARSING_ELEMENT;
	eResult = pElement->QueryIntAttribute("polygonNum", &polygon_num);
	XMLElement * pPolygonElement = pElement->FirstChildElement("Polygon");	
	polygons_.resize(polygon_num);
	for (int i = 0; i < polygon_num; i++) {
		if (pPolygonElement == nullptr)
			return XML_ERROR_PARSING_ELEMENT;
		int point_num;
		eResult = pPolygonElement->QueryIntAttribute("pointNum", &point_num);
		XMLElement * pPointElement = pPolygonElement->FirstChildElement("Point");		
		vector<PPOINTF> points(point_num);
		for (int j = 0; j < point_num; j++) {
			eResult = pPointElement->QueryFloatAttribute("x", &points[j].x);
			eResult = pPointElement->QueryFloatAttribute("y", &points[j].y);
			pPointElement = pPointElement->NextSiblingElement("Point");
		}
		RoadMarkerInfo info;
		int road_marker_type = 0;
		eResult = pPolygonElement->QueryIntAttribute("type", &road_marker_type);
		switch (road_marker_type)
		{
		case 0:
			polygons_[i].SetRoadMarkType(RoadMarkerInfo::ROAD_MARKER_TYPE_HARD_NEGATIVE);
			break;
		case 1:
			polygons_[i].SetRoadMarkType(RoadMarkerInfo::ROAD_MARKER_TYPE_STOP_LINE);
			break;
		case 2:
			polygons_[i].SetRoadMarkType(RoadMarkerInfo::ROAD_MARKER_TYPE_CROSSWALK);
			break;
		case 3:
			polygons_[i].SetRoadMarkType(RoadMarkerInfo::ROAD_MARKER_TYPE_ARROW);
			break;
		case 4:
			polygons_[i].SetRoadMarkType(RoadMarkerInfo::ROAD_MARKER_TYPE_SPEED_BUMP);
			break;
		default:
			polygons_[i].SetRoadMarkType(RoadMarkerInfo::ROAD_MARKER_TYPE_HARD_NEGATIVE);
			break;
		}
		polygons_[i].SetPoints(points);
		pPolygonElement = pPolygonElement->NextSiblingElement("Polygon");
	}

	pElement = pRoot->FirstChildElement("Boundarys");
	if (pElement == NULL)
		return XML_ERROR_PARSING_ELEMENT;
	int boundary_num = 0, boundary_count = 0;
	eResult = pElement->QueryIntAttribute("boundaryNum", &boundary_num);

	XMLElement * pBoundaryElement = pElement->FirstChildElement("Boundary");
	boundarys_.resize(boundary_num);
	for (int i = 0; i < boundary_num; i++) {
		if (pBoundaryElement == nullptr)
			return XML_ERROR_PARSING_ELEMENT;

		int point_num;
		int type3, boundary_id;
		int boundary_unknown_id = 8;
		BoundaryInfo::CATEGORY3 position = BoundaryInfo::CATEGORY3::C3_NONE;
		double sum_x = 0.F, average_x = 0.F;

		eResult = pBoundaryElement->QueryIntAttribute("type3", &type3);
		eResult = pBoundaryElement->QueryIntAttribute("boundary", &boundarys_[i].info.BoundaryType);
		eResult = pBoundaryElement->QueryIntAttribute("pointNum", &point_num);
		eResult = pBoundaryElement->QueryIntAttribute("occNum", &boundary_count);

		XMLElement * pPointElement = pBoundaryElement->FirstChildElement("Point");
		boundarys_[i].spline_x_.resize(point_num);
		boundarys_[i].spline_y_.resize(point_num);
		boundarys_[i].line_r_.resize(point_num);
		for (int j = 0; j < point_num; j++) {
			eResult = pPointElement->QueryDoubleAttribute("x", &boundarys_[i].spline_x_[j]);
			eResult = pPointElement->QueryDoubleAttribute("y", &boundarys_[i].spline_y_[j]);
			eResult = pPointElement->QueryDoubleAttribute("r", &boundarys_[i].line_r_[j]);
			boundarys_[i].line_r_[j] = DEFUALT_BOUNDARY_W;
			pPointElement = pPointElement->NextSiblingElement("Point");
		}

		boundarys_[i].info.occlusions_top_bottom_.resize(boundary_count);
		XMLElement * pOccElement = pBoundaryElement->FirstChildElement("Occlusion");   // 가려짐
		for (int k = 0; k < boundary_count; k++) {
			float top_y, bottom_y;

			eResult = pOccElement->QueryFloatAttribute("top", &top_y);
			eResult = pOccElement->QueryFloatAttribute("bottom", &bottom_y);

			boundarys_[i].info.occlusions_top_bottom_[k] = std::make_pair(top_y, bottom_y);
			pOccElement = pOccElement->NextSiblingElement("Occlusion");
		}

		boundary_id = (type3 >> 8);
		position = static_cast<BoundaryInfo::CATEGORY3>(type3 % (1 << 8));
		switch (position)
		{
		case BoundaryInfo::CATEGORY3::LEFT:
		case BoundaryInfo::CATEGORY3::RIGHT:
			boundarys_[i].info.SetType3(position, boundary_id);
			break;
		default:
			sum_x += boundarys_[i].spline_x_[0];
			sum_x += boundarys_[i].spline_x_[point_num - 1];

			average_x = sum_x / 2;

			if (average_x < image_width_ / 2)
				position = BoundaryInfo::CATEGORY3::LEFT;
			else 
				position = BoundaryInfo::CATEGORY3::RIGHT;

			boundarys_[i].info.SetType3(position, boundary_unknown_id);
			break;
		}
		boundarys_[i].GenerateModels();
		pBoundaryElement = pBoundaryElement->NextSiblingElement("Boundary");
	}
	return true;
}


bool RoadLaneManager::WriteFile(const char *szpath)
{
	tinyxml2::XMLDocument xmlDoc;
	
	XMLNode * pRoot = xmlDoc.NewElement("RoadLane");
	xmlDoc.InsertFirstChild(pRoot);

	XMLElement *pElement = xmlDoc.RootElement();
	pElement->SetAttribute("toolVersion", str_tool_version_.c_str());     // Tool version
	pElement->SetAttribute("imageWidth", image_width_);     // 이미지 가로 크기
	pElement->SetAttribute("imageHeight", image_height_);   // 이미지 세로 크기
	
	pElement = xmlDoc.NewElement("VP"); // VP
	pElement->SetAttribute("hasVP", has_vp_);
	pElement->SetAttribute("y_ratio", vp_y_ratio_);
	pElement->SetAttribute("x_ratio", vp_x_ratio_);
	pRoot->InsertEndChild(pElement);				 // VP 추가

	pElement = xmlDoc.NewElement("Splines");		 // Splines	
	for (int i = 0; i < lines_.size(); i++){
		XMLElement * pSplineElement = xmlDoc.NewElement("Spline");     // Spline
		pSplineElement->SetAttribute("type1", lines_[i].info.type1);
		pSplineElement->SetAttribute("type2", lines_[i].info.type2);
		pSplineElement->SetAttribute("type3", lines_[i].info.type3);
		pSplineElement->SetAttribute("type4", lines_[i].info.type4);
		pSplineElement->SetAttribute("type5", lines_[i].info.type5);
		pSplineElement->SetAttribute("type6", lines_[i].info.type6);
		pSplineElement->SetAttribute("pointNum", (int)lines_[i].spline_x_.size());
		pSplineElement->SetAttribute("occNum", (int)lines_[i].info.occlusions_top_bottom_.size());

		for (int j = 0; j < lines_[i].spline_x_.size(); j++) {
			XMLElement * pPointElement = xmlDoc.NewElement("Point");    // Spline의 한 점
			pPointElement->SetAttribute("x", lines_[i].spline_x_[j]);
			pPointElement->SetAttribute("y", lines_[i].spline_y_[j]);
			pPointElement->SetAttribute("r", lines_[i].line_r_[j]);

			pSplineElement->InsertEndChild(pPointElement);              // Spline의 한 점 추가
		}

		for (int k = 0; k < lines_[i].info.occlusions_top_bottom_.size(); k++) {
			XMLElement * pOccElement = xmlDoc.NewElement("Occlusion");   // 가려짐
			pOccElement->SetAttribute("top", lines_[i].info.occlusions_top_bottom_[k].first);
			pOccElement->SetAttribute("bottom", lines_[i].info.occlusions_top_bottom_[k].second);

			pSplineElement->InsertEndChild(pOccElement);				 // 가려짐 추가
		}

		pElement->InsertEndChild(pSplineElement);				  	  // Spline 추가
	}

	pElement->SetAttribute("splineNum", (int)lines_.size());
	pRoot->InsertEndChild(pElement);				  // Splines 추가

	pElement = xmlDoc.NewElement("Polygons");		 // Polygons
	for (int i = 0; i < polygons_.size(); i++) {
		XMLElement * pPolygonElement = xmlDoc.NewElement("Polygon");     // Polygon
		pPolygonElement->SetAttribute("pointNum", polygons_[i].GetPolygonPointNum());
		for (int j = 0; j < polygons_[i].GetPolygonPointNum(); j++) {
			XMLElement * pPointElement = xmlDoc.NewElement("Point");    // Spline의 한 점
			pPointElement->SetAttribute("x", polygons_[i].GetPoint(j).x);
			pPointElement->SetAttribute("y", polygons_[i].GetPoint(j).y);

			pPolygonElement->InsertEndChild(pPointElement);              // Spline의 한 점 추가
		}
		pPolygonElement->SetAttribute("type", polygons_[i].GetRoadMarkerInfo().GetType());
		pElement->InsertEndChild(pPolygonElement);				  	  // Spline 추가
	}

	pElement->SetAttribute("polygonNum", (int)polygons_.size());
	pRoot->InsertEndChild(pElement);
	
	pElement = xmlDoc.NewElement("Boundarys");		 // Boundarys	
	for (int i = 0; i < boundarys_.size(); i++) {
		XMLElement * pBoundaryElement = xmlDoc.NewElement("Boundary");     // Boundary
		pBoundaryElement->SetAttribute("type3", boundarys_[i].info.type3);
		pBoundaryElement->SetAttribute("boundary", boundarys_[i].info.BoundaryType);
		pBoundaryElement->SetAttribute("pointNum", (int)boundarys_[i].spline_x_.size());
		pBoundaryElement->SetAttribute("occNum", (int)boundarys_[i].info.occlusions_top_bottom_.size());

		for (int j = 0; j < boundarys_[i].spline_x_.size(); j++) {
			XMLElement * pPointElement = xmlDoc.NewElement("Point");    // Spline의 한 점
			pPointElement->SetAttribute("x", boundarys_[i].spline_x_[j]);
			pPointElement->SetAttribute("y", boundarys_[i].spline_y_[j]);
			pPointElement->SetAttribute("r", DEFUALT_BOUNDARY_W);

			pBoundaryElement->InsertEndChild(pPointElement);              // Spline의 한 점 추가
		}

		for (int k = 0; k < boundarys_[i].info.occlusions_top_bottom_.size(); k++) {
			XMLElement * pOccElement = xmlDoc.NewElement("Occlusion");   // 가려짐
			pOccElement->SetAttribute("top", boundarys_[i].info.occlusions_top_bottom_[k].first);
			pOccElement->SetAttribute("bottom", boundarys_[i].info.occlusions_top_bottom_[k].second);

			pBoundaryElement->InsertEndChild(pOccElement);				 // 가려짐 추가
		}

		pElement->InsertEndChild(pBoundaryElement);				  	  // Spline 추가
	}

	pElement->SetAttribute("boundaryNum", (int)boundarys_.size());
	pRoot->InsertEndChild(pElement);				  // Splines 추가

	XMLError eResult = xmlDoc.SaveFile(szpath); // 파일 저장
	if (eResult != XMLError::XML_SUCCESS)
	{
		//printf("파일 저장 실패.\n");
		return 0;
	}
	return true;
}
void RoadLaneManager::Reset(int image_width, int image_height)
{	
	image_width_ = image_width;
	image_height_ = image_height;
	polygons_.clear();
	lines_.clear();
	boundarys_.clear();
	vp_y_ratio_ = DEFUALT_VP_Y;
	has_vp_ = false;
}
void RoadLaneManager::AddLine(const LaneLine &line)
{
	lines_.push_back(line);
}
void RoadLaneManager::ResetLine(const LaneLine &line, int idx)
{
	if(idx >= 0 && idx < lines_.size())
		lines_[idx] = line;
}
void RoadLaneManager::RemoveLine(int idx)
{
	if(idx >= 0 && idx < lines_.size()){
		lines_.erase(lines_.begin() + idx);
	}
}
void RoadLaneManager::AddBoundary(const BoundaryLine &line)
{
	boundarys_.push_back(line);
}
void RoadLaneManager::ResetBoundary(const BoundaryLine &line, int idx)
{
	if (idx >= 0 && idx < boundarys_.size())
		boundarys_[idx] = line;
}
void RoadLaneManager::RemoveBoundary(int idx)
{
	if (idx >= 0 && idx < boundarys_.size()) {
		boundarys_.erase(boundarys_.begin() + idx);
	}
}

void RoadLaneManager::AddPolygon(const RoadMarkingPolygon &polygon)
{
	polygons_.push_back(polygon);
}
void RoadLaneManager::ResetPoygon(const RoadMarkingPolygon &polygon, int idx)
{
	if (idx >= 0 && idx < polygons_.size())
		polygons_[idx] = polygon;
}
void RoadLaneManager::RemovePoygon(int idx)
{
	if (idx >= 0 && idx < polygons_.size()) {
		polygons_.erase(polygons_.begin() + idx);
	}
}

void RoadLaneManager::SetVPYRatio(double vp_y_ratio)
{
	has_vp_ = true;
	vp_y_ratio_ = vp_y_ratio;
}

void RoadLaneManager::SetVPXRatio(double vp_x_ratio)
{
	vp_x_ratio_ = vp_x_ratio;
}

bool SortLane(LaneLine &p1, LaneLine &p2){
	return (p1.bottom_y_ - p1.top_y_) < (p2.bottom_y_ - p2.top_y_);
}

void RoadLaneManager::SortingLength(){
	std::sort(lines_.begin(), lines_.end(), SortLane);
}

void RoadLaneManager::SetImageSize(int width, int height)
{
	image_width_ = width;
	image_height_ = height;
}

void RoadLaneManager::SetToolVersion(string str_version)
{
	str_tool_version_ = str_version;
}

