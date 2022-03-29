#include "stdafx.h"
#include "PointingToolView.h"
#include "regressor.h"
#include <io.h>
#include <stack>

bool comp(PPOINTF &a, PPOINTF &b) {
	return (a.y < b.y);
}

bool OcclusionStartYConstraint(LaneLine &line, float y, std::pair<float, float> &constTopBottom) {
	//1. 
	if (line.top_y_ > y) { return false; }
	if (line.bottom_y_ < y) { return false; }
	for (int i = 0; i < line.info.occlusions_top_bottom_.size(); i++) {
		if (line.info.occlusions_top_bottom_[i].first <= y &&
			line.info.occlusions_top_bottom_[i].second >= y) {
			return false;
		}
	}

	float top_y = line.top_y_;
	float bottom_y = line.bottom_y_;

	float min_dist_top = 100000.;
	float min_dist_bottom = 100000.;

	for (int i = 0; i < line.info.occlusions_top_bottom_.size(); i++) {
		if (line.info.occlusions_top_bottom_[i].first - y > 0 &&
			min_dist_top > line.info.occlusions_top_bottom_[i].first - y) {
			min_dist_top = line.info.occlusions_top_bottom_[i].first - y;
			bottom_y = line.info.occlusions_top_bottom_[i].first;
		}
		if (y - line.info.occlusions_top_bottom_[i].first > 0 &&
			min_dist_bottom > y - line.info.occlusions_top_bottom_[i].first) {
			min_dist_bottom = y - line.info.occlusions_top_bottom_[i].first;
			top_y = line.info.occlusions_top_bottom_[i].first;
		}
	}

	constTopBottom = make_pair(top_y, bottom_y);

	return true;
}



bool OCCControlRegion(LaneLine &line, PPOINTF img_point, int &idx, int &top0_bottom1) {
	for (int i = 0; i < line.info.occlusions_top_bottom_.size(); i++) {
		PPOINT3F point = line.EstimatePoint(line.info.occlusions_top_bottom_[i].first);
		RectF rect(point.x - point.r, point.y - point.r, point.r * 2, point.r * 2);
		if (rect.Contains(img_point.x, img_point.y)) {
			idx = i;
			top0_bottom1 = false;
			return true;
		}
	}
	for (int i = 0; i < line.info.occlusions_top_bottom_.size(); i++) {
		PPOINT3F point = line.EstimatePoint(line.info.occlusions_top_bottom_[i].second);
		RectF rect(point.x - point.r, point.y - point.r, point.r * 2, point.r * 2);
		if (rect.Contains(img_point.x, img_point.y)) {
			idx = i;
			top0_bottom1 = true;
			return true;
		}
	}
	return false;
}

int CPointingToolView::MousePointOnSplineOcclusion(LaneLine &line, const CPoint &point) {
	PPOINTF view_pt(point.x, point.y);
	PPOINTF image_pt = mapv2i(view_pt);
	for (int i = 0; i < line.info.occlusions_top_bottom_.size(); i++) {
		if (line.info.occlusions_top_bottom_[i].first < image_pt.y &&
			line.info.occlusions_top_bottom_[i].second > image_pt.y) {
			return i;
		}
	}
	return -1;
}

void CPointingToolView::MouseMoveSpline(const CPoint &point)
{
	m_bOccControlPoint = false;
	m_bMouseOnVP = false;
	PPOINTF ipt = mapv2i(PPOINTF(point.x, point.y));
	if (m_bLDownVP) {
		m_RoadLane.SetVPXRatio(ipt.x / m_pImage->GetWidth());
		m_RoadLane.SetVPYRatio(ipt.y / m_pImage->GetHeight());
		m_pNaviWnd->Invalidate(FALSE);
	}
	else {
		int sel_spline = MousePointOnSpline(point);
		if (m_bLDownOcclusion == false) {
			if (sel_spline != -1) {
				m_nSplineMouseOver = sel_spline;
			}
			else {
				m_nSplineMouseOver = -1;
			}
		}

		if (m_bCheckOcclusion) {
			if (m_nSplineMouseOver != -1)
				m_nOccMouseOver = MousePointOnSplineOcclusion(*m_RoadLane.line_ptr(m_nSplineMouseOver), point);
			int top0_bottom1;
			int idx;
			if (m_bLDownOcclusion) {
				if (occConstTopBottom.first < ipt.y && occConstTopBottom.second > ipt.y) {
					occEnd = m_RoadLane.line_ptr(m_nSplineMouseOver)->EstimatePoint(ipt.y);
				}
			}
			else if (m_nSplineMouseOver != -1 &&
				OCCControlRegion(*m_RoadLane.line_ptr(m_nSplineMouseOver), ipt, idx, top0_bottom1)) {
				m_bOccControlPoint = true;
			}
		}
		m_CurrentPoint.point.x = ipt.x;
		m_CurrentPoint.point.y = ipt.y;
		if (m_CurrentPoint.regression_r) m_CurrentPoint.point.r = m_WorkingSplineLine.reg_ir(ipt.y);
		if (m_bControlSplinePoint) {
			if (m_nSelSplinePoint >= 0 && m_nSelSplinePoint < m_WorkingSplineLine.element_size()) {
				PPOINT3F pt = m_WorkingSplineLine.ixyr(m_nSelSplinePoint);
				pt.x = ipt.x;
				pt.y = ipt.y;
				m_WorkingSplineLine.set_ixyr(pt, m_nSelSplinePoint);
			}
		}
		else if (m_bLDownVP == false && m_nSplineMouseOver == -1) {
			if (MousePointOnVP(point)) {
				m_bMouseOnVP = true;
			}
		}
	}
}
void CPointingToolView::LButtonDownSpline(const CPoint &point)
{
	int sel_spline = MousePointOnSpline(point);

	if (m_bCheckOcclusion) {
		LaneLine *line = m_RoadLane.line_ptr(sel_spline);
		if (line) {
			PPOINTF ipt = mapv2i(PPOINTF(point.x, point.y));
			if (m_bOccControlPoint) {
				int idx;
				int top0_bottom1;
				if (OCCControlRegion(*m_RoadLane.line_ptr(m_nSplineMouseOver), ipt, idx, top0_bottom1)) {
					PPOINT3F pt1 = m_RoadLane.line_ptr(m_nSplineMouseOver)->EstimatePoint(
						m_RoadLane.line_ptr(m_nSplineMouseOver)->info.occlusions_top_bottom_[idx].first);
					PPOINT3F pt2 = m_RoadLane.line_ptr(m_nSplineMouseOver)->EstimatePoint(
						m_RoadLane.line_ptr(m_nSplineMouseOver)->info.occlusions_top_bottom_[idx].second);

					if (top0_bottom1 == 0) {
						occEnd = pt1;
						occStart = pt2;
					}
					else {
						occEnd = pt2;
						occStart = pt1;
					}
					m_RoadLane.line_ptr(m_nSplineMouseOver)->info.occlusions_top_bottom_.erase(
						m_RoadLane.line_ptr(m_nSplineMouseOver)->info.occlusions_top_bottom_.begin() + idx);
					m_bLDownOcclusion = true;
					OcclusionStartYConstraint(*line, occEnd.y, occConstTopBottom);
				}
			}
			else if (OcclusionStartYConstraint(*line, ipt.y, occConstTopBottom)) {
				m_bLDownOcclusion = true;
				occStart = occEnd = m_RoadLane.line_ptr(m_nSplineMouseOver)->EstimatePoint(ipt.y);
				occEnd = occStart;
			}
		}
		else if (m_bVisibleVP == true && m_bMouseOnVP) {
			m_bLDownVP = true;
		}
	}
	else {
		if (sel_spline != -1) {
			m_nSelSpline = sel_spline;
			m_WorkingSplineLine.Reset();
			LaneLine &line = *m_RoadLane.line_ptr(m_nSelSpline);
			for (int i = 0; i < line.spline_x_.size(); i++) {
				PPOINT3F ipoint3(line.spline_x_[i], line.spline_y_[i], line.line_r_[i]);
				m_WorkingSplineLine.AddPoint3(ipoint3);
			}
			m_WorkingSplineLine.GenerateModels();
			m_WorkingSplineLine.info = line.info;
		}
		else {
			int idx = MousePointOnSplinePoint(point);
			/*if(m_bLbuttonControlPoint){
			m_bLbuttonControlPoint = false;
			}*/
			if (idx != -1) {
				m_nSelSplinePoint = idx;
			}
			else if (m_bVisibleVP == true && m_bMouseOnVP) {
				m_bLDownVP = true;
			}
			else {
				PPOINT3F point3 = mapv2i(PPOINT3F(point));
				point3.r = m_WorkingSplineLine.reg_ir(point3.y);
				m_WorkingSplineLine.AddPoint3(point3);
				m_nSelSplinePoint = m_WorkingSplineLine.element_size() - 1;
				m_WorkingSplineLine.GenerateModels();
				//m_bLbuttonControlPoint = true;
			}
		}
	}

	//DrawSplines(point);
}

void CPointingToolView::LButtonUpSpline(const CPoint &point) {
	m_bLDownVP = false;
	if (m_bLDownOcclusion) {
		PPOINTF ipt = mapv2i(PPOINTF(point.x, point.y));
		LaneLine &line = *m_RoadLane.line_ptr(m_nSplineMouseOver);
		if (occConstTopBottom.first < ipt.y && occConstTopBottom.second > ipt.y) {
			occEnd = m_RoadLane.line_ptr(m_nSplineMouseOver)->EstimatePoint(ipt.y);
		}
		if (abs(occStart.y - occEnd.y) > 1)
			line.info.occlusions_top_bottom_.push_back(std::make_pair(std::min(occStart.y, occEnd.y), std::max(occStart.y, occEnd.y)));

		m_bLDownOcclusion = false;
		//DrawSplines(point);
	}
	if (m_bRbuttonControlPoint == true && m_bControlSplinePoint == true) {
		m_bRbuttonControlPoint = false;
		m_bControlSplinePoint = false;
		//DrawSplines(point);
	}/* else if(m_bLbuttonControlPoint){
		m_bControlSplinePoint = true;
		DrawSplines(point);
	} else if(m_bLbuttonControlPoint == false){
		m_bControlSplinePoint = false;
		DrawSplines(point);
	}*/


}
bool CPointingToolView::RButtonDownSpline(const CPoint &point)
{

	return false;
}
void CPointingToolView::RButtonUpSpline(const CPoint &point)
{
	CPoint point_pt = point;
	if (m_RButtonClickPrevPt == point) {
		int sel_point = MousePointOnSplinePoint(point);
		if (sel_point > -1 && m_bControlSplinePoint == false) {
			m_nSelSplinePoint = sel_point;
			m_bControlSplinePoint = true;
			m_bRbuttonControlPoint = true;
			//point_pt.x = m_WorkingSplineLine.vx(sel_point);
			//point_pt.y = m_WorkingSplineLine.vy(sel_point);
		}
		else if (m_bControlSplinePoint) {
			m_bControlSplinePoint = false;
			m_bRbuttonControlPoint = false;
		}
		else {
			if (m_WorkingSplineLine.element_size() == 0) {  // 아래의 if문과 순서 수정 - 커서 위치에 따라 type분류창 열릴 수 있음 (else if도 가능)
				m_WorkingSplineLine.Reset();
				m_nSelSplinePoint = -1;
				m_nSelSpline = -1;

				int sel_id = MousePointOnSpline(point);
				if (sel_id != -1) {
					CSplineManagerDlg dlg;
					dlg.m_spline = m_RoadLane.line_ptr(sel_id);
					CPoint pos = point;
					ClientToScreen(&pos);
					dlg.window_pos = pos;

					CRect rect;
					GetClientRect(rect);
					dlg.left_oriented = rect.Width() * 0.5 < point.x ? false : true;
					dlg.DoModal();

					if (dlg.m_spline->info.GetType2() == LaneInfo::SINGLE
						&& dlg.m_spline->info.GetType3() != LaneInfo::UNCERTAIN
						&& dlg.m_spline->info.GetType4() == LaneInfo::C4_NONE) {
						for (int i = 0; i < m_RoadLane.GetSizeLaneLine(); i++) {
							if (i == sel_id)
								continue;
							if (m_RoadLane.line(i).info.GetType3() == dlg.m_spline->info.GetType3()
								&& m_RoadLane.line(i).info.GetType3_ID() == dlg.m_spline->info.GetType3_ID()
								&& m_RoadLane.line(i).info.GetType2() == dlg.m_spline->info.GetType2()
								&& m_RoadLane.line(i).info.GetType4() == dlg.m_spline->info.GetType4()) {
								//AfxMessageBox(_T("타입 3가 중복되었습니다."));
							}
						}
					}
				}
			}

			if (m_WorkingSplineLine.element_size()) {
				m_WorkingSplineLine.erase_last_one();
				m_nSelSplinePoint--;
			}

			m_bControlSplinePoint = false;
			m_bLbuttonControlPoint = false;
		}
	}
	//DrawSplines(point_pt);
}

bool CPointingToolView::MouseWheelSpline(UINT nFlags, short zDelta, CPoint pt)
{
	if (m_WorkingSplineLine.element_size() == 0) return false;

	float scale = g_view_scale;
	if (scale < 1) {
		scale = 1;
	}
	else {
		scale = 9.3633333333333500e-001
			+ 9.9201631701631720e-002  * scale
			+ -4.2144522144522314e-002  * scale * scale
			+ 2.4417249417249580e-003 * scale * scale * scale;
	}
	float zoom_gain = scale;

	::ScreenToClient(this->m_hWnd, &pt);
	int idx = MousePointOnSplinePoint(pt);
	if (idx != -1 && idx == m_nSelSplinePoint && m_bControlSplinePoint == true) {
		if (zDelta > 0) {
			if (m_nSelSplinePoint >= 0 && m_nSelSplinePoint < m_WorkingSplineLine.element_size()) {
				PPOINT3F pt = m_WorkingSplineLine.ixyr(m_nSelSplinePoint);
				pt.r += zoom_gain;
				m_WorkingSplineLine.set_ixyr(pt, m_nSelSplinePoint);
			}
		}
		else if (zDelta < 0) {
			PPOINT3F pt = m_WorkingSplineLine.ixyr(m_nSelSplinePoint);
			pt.r -= zoom_gain;
			pt.r = std::max(pt.r, 0.5f);
			m_WorkingSplineLine.set_ixyr(pt, m_nSelSplinePoint);
		}
		return true;
	}
	return false;
}

bool CPointingToolView::KeyUpSpline(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	if (nChar == 90) {   // z
		/*if(m_bShowSplineOnlySupport == true){

			m_bShowSplineOnly = !m_bShowSplineOnly;
			m_bShowSplineOnlySupport = false;
			POINT p;
			GetCursorPos(&p);
			::ScreenToClient(this->m_hWnd, &p);
			DrawSplines(p);
		}*/
	}

	if (nChar == 88) {    // x
		if (m_bControlSplinePoint) {
			m_bControlSplinePoint = false;
			m_bRbuttonControlPoint = false;
			return true;
		}
	}

	return false;
}

bool CPointingToolView::KeyDownSpline(UINT nChar, UINT nRepCnt, UINT nFlags) {

	float scale = g_view_scale;
	if (scale < 1) {
		scale = 1;
	}
	else {
		scale = 9.3633333333333500e-001
			+ 9.9201631701631720e-002  * scale
			+ -4.2144522144522314e-002  * scale * scale
			+ 2.4417249417249580e-003 * scale * scale * scale;
	}
	float zoom_gain = scale;
	/*if(nChar == 90){ //z - 스플라인만 보이기
		m_bShowSplineOnly = !m_bShowSplineOnly;
		g_pToolView->CheckDlgButton(IDC_LANELINE_SHOW_SPLINE_ONLY, m_bShowSplineOnly);
		//g_pToolView->UpdateData(FALSE);
		SetFocus();
		return true;
	}*/
	/*if(nChar == VK_F1){ // F1 - 가려짐 체크
		m_bCheckOcclusion = !m_bCheckOcclusion;
		g_pToolView->CheckDlgButton(IDC_LANELINE_CHECK_OCCLUSION, m_bCheckOcclusion);
		//g_pToolView->UpdateData(FALSE);
		POINT p;
		GetCursorPos(&p);
		::ScreenToClient(this->m_hWnd, &p);
		DrawSplines(p);
		SetFocus();
	}*/
	///////////////////////// 추가 ////////////////////////////////
	/*if (nChar == VK_F1) { // F1 - 가려짐
	g_pToolView->CheckDlgButton(IDC_LANELINE_CHECK_OCCLUSION, !g_pToolView->IsDlgButtonChecked(IDC_LANELINE_CHECK_OCCLUSION));
	g_pToolView -> OnBnClickedLanelineCheckOcclusion();
	}*/
	/*if (nChar == VK_F2) { // F2 - 소실점 체크
		g_pToolView->CheckDlgButton(IDC_LANELINE_VP, !g_pToolView->IsDlgButtonChecked(IDC_LANELINE_VP));
		g_pToolView->OnBnClickedLanelineVp();
	}
	if (nChar == VK_F3) { // F3 - 소실점 보이기
		g_pToolView->CheckDlgButton(IDC_VP_VISIBLE, !g_pToolView->IsDlgButtonChecked(IDC_VP_VISIBLE));
		g_pToolView->OnBnClickedVpVisible();
	}*/
	///////////////////////////////////////////////////////////////


	if (nChar == 88) {//x		
		if (m_bControlSplinePoint == false && m_nSelSplinePoint >= 0 &&
			m_nSelSplinePoint < m_WorkingSplineLine.element_size() &&
			m_bControlSplinePoint == false) {
			POINT p;
			GetCursorPos(&p);
			::ScreenToClient(this->m_hWnd, &p);
			int sel_point = MousePointOnSplinePoint(p);
			if (sel_point == m_nSelSplinePoint) {
				m_bControlSplinePoint = true;
				m_bRbuttonControlPoint = true;
				return true;		// x 처음 눌렀을 때만 true 반환하여 화면 업데이트(사각형 제거)
			}
		}
		return false;				// spline출력이 느려지는 오류 제거 - Why??
	}

	if (nChar == 67) { //c		// overlay값 감소
		int ov_pos = g_pToolView->m_OvlySlider.GetPos() - 5;
		g_pToolView->m_OvlySlider.SetPos(std::max(ov_pos, 0));
		SetFocus();
		return true;
	}

	if (nChar == 86) {//v			// overlay값 증가
		int ov_pos = g_pToolView->m_OvlySlider.GetPos() + 5;
		g_pToolView->m_OvlySlider.SetPos(std::min(ov_pos, 255));
		SetFocus();
		return true;
	}

	if (nChar == VK_ESCAPE) {
		m_nSelSpline = -1;
		m_nSelSplinePoint = -1;
		m_bControlSplinePoint = false;
		m_WorkingSplineLine.Reset();
		return true;
	}
#ifndef INSPECTION
	if (m_nSelSplinePoint >= 0 && m_nSelSplinePoint < m_WorkingSplineLine.element_size()) {
		if (nChar == VK_UP) {
			PPOINT3F pt = m_WorkingSplineLine.ixyr(m_nSelSplinePoint);
			pt.y -= zoom_gain;
			m_WorkingSplineLine.set_ixyr(pt, m_nSelSplinePoint);
			SetCursorPosSelectedPoint();
			return true;
		}
		else if (nChar == VK_DOWN) {
			PPOINT3F pt = m_WorkingSplineLine.ixyr(m_nSelSplinePoint);
			pt.y += zoom_gain;
			m_WorkingSplineLine.set_ixyr(pt, m_nSelSplinePoint);
			SetCursorPosSelectedPoint();
			return true;
		}
		else if (nChar == VK_LEFT) {
			PPOINT3F pt = m_WorkingSplineLine.ixyr(m_nSelSplinePoint);
			pt.x -= zoom_gain;
			m_WorkingSplineLine.set_ixyr(pt, m_nSelSplinePoint);
			SetCursorPosSelectedPoint();
			return true;
		}
		else if (nChar == VK_RIGHT) {
			PPOINT3F pt = m_WorkingSplineLine.ixyr(m_nSelSplinePoint);
			pt.x += zoom_gain;
			m_WorkingSplineLine.set_ixyr(pt, m_nSelSplinePoint);
			SetCursorPosSelectedPoint();
			return true;
		}
	}
	if (m_WorkingSplineLine.spline_y_.size()) {
		if (nChar == VK_UP) {
			for (int i = 0; i < m_WorkingSplineLine.spline_y_.size(); i++) {
				PPOINT3F pt = m_WorkingSplineLine.ixyr(i);
				pt.y -= zoom_gain;
				m_WorkingSplineLine.set_ixyr(pt, i);
			}
			return true;
		}
		else if (nChar == VK_DOWN) {
			for (int i = 0; i < m_WorkingSplineLine.spline_y_.size(); i++) {
				PPOINT3F pt = m_WorkingSplineLine.ixyr(i);
				pt.y += zoom_gain;
				m_WorkingSplineLine.set_ixyr(pt, i);
			}
			return true;
		}
		else if (nChar == VK_LEFT) {
			for (int i = 0; i < m_WorkingSplineLine.spline_y_.size(); i++) {
				PPOINT3F pt = m_WorkingSplineLine.ixyr(i);
				pt.x -= zoom_gain;
				m_WorkingSplineLine.set_ixyr(pt, i);
			}
			return true;
		}
		else if (nChar == VK_RIGHT) {
			for (int i = 0; i < m_WorkingSplineLine.spline_y_.size(); i++) {
				PPOINT3F pt = m_WorkingSplineLine.ixyr(i);
				pt.x += zoom_gain;
				m_WorkingSplineLine.set_ixyr(pt, i);
			}
			return true;
		}
	}

#endif
	if (nChar == VK_ADD) {
		if (m_nSelSplinePoint >= 0 && m_nSelSplinePoint < m_WorkingSplineLine.element_size()) {
			PPOINT3F pt = m_WorkingSplineLine.ixyr(m_nSelSplinePoint);
			pt.r += zoom_gain;
			m_WorkingSplineLine.set_ixyr(pt, m_nSelSplinePoint);
		}
		return true;
	}
	else if (nChar == VK_SUBTRACT) {
		if (m_nSelSplinePoint >= 0 && m_nSelSplinePoint < m_WorkingSplineLine.element_size()) {
			PPOINT3F pt = m_WorkingSplineLine.ixyr(m_nSelSplinePoint);
			pt.r -= zoom_gain;
			pt.r = std::max(pt.r, 0.5f);
			m_WorkingSplineLine.set_ixyr(pt, m_nSelSplinePoint);
		}
		return true;
	}
	else if (nChar == _T('d') || nChar == _T('D')) {
		if (m_nSelSplinePoint >= 0 && m_nSelSplinePoint < m_WorkingSplineLine.element_size()) {
			PPOINT3F pt = m_WorkingSplineLine.ixyr(m_nSelSplinePoint);
			pt.r += zoom_gain;
			m_WorkingSplineLine.set_ixyr(pt, m_nSelSplinePoint);
		}
		return true;
	}
	else if (nChar == _T('s') || nChar == _T('S')) {
		if (m_nSelSplinePoint >= 0 && m_nSelSplinePoint < m_WorkingSplineLine.element_size()) {
			PPOINT3F pt = m_WorkingSplineLine.ixyr(m_nSelSplinePoint);
			pt.r -= zoom_gain;
			pt.r = std::max(pt.r, 0.5f);
			m_WorkingSplineLine.set_ixyr(pt, m_nSelSplinePoint);
		}
		return true;
	}
	else if (nChar == _T('a') || nChar == _T('A')) {
		if (m_WorkingSplineLine.element_size() == 0) {  // 아래의 if문과 순서 수정 - 커서 위치에 따라 type분류창 열릴 수 있음 (else if도 가능)
			m_WorkingSplineLine.Reset();
			m_nSelSplinePoint = -1;
			m_nSelSpline = -1;

			POINT point;
			GetCursorPos(&point);
			::ScreenToClient(this->m_hWnd, &point);

			int sel_id = MousePointOnSpline(point);
			if (sel_id != -1) {
				CSplineManagerDlg dlg;
				dlg.m_spline = m_RoadLane.line_ptr(sel_id);
				CPoint pos = point;
				ClientToScreen(&pos);
				dlg.window_pos = pos;

				CRect rect;
				GetClientRect(rect);
				dlg.left_oriented = rect.Width() * 0.5 < point.x ? false : true;
				dlg.DoModal();

				if (dlg.m_spline->info.GetType2() == LaneInfo::SINGLE
					&& dlg.m_spline->info.GetType3() != LaneInfo::UNCERTAIN
					&& dlg.m_spline->info.GetType4() != LaneInfo::OPPOSITE_SIDE) {
					for (int i = 0; i < m_RoadLane.GetSizeLaneLine(); i++) {
						if (i == sel_id)
							continue;
						if (m_RoadLane.line(i).info.GetType3() == dlg.m_spline->info.GetType3()
							&& m_RoadLane.line(i).info.GetType3_ID() == dlg.m_spline->info.GetType3_ID()
							&& m_RoadLane.line(i).info.GetType2() == dlg.m_spline->info.GetType2()
							&& m_RoadLane.line(i).info.GetType4() == dlg.m_spline->info.GetType4()) {
							AfxMessageBox(_T("타입 3가 중복되었습니다."));
						}
					}
				}
			}
		}
		return true;
	}
	//else if(nChar == VK_TAB){
	//	m_nSelSplinePoint--;
	//	if(m_nSelSplinePoint < 0){
	//		m_nSelSplinePoint = m_WorkingSplineLine.element_size() - 1;
	//	}
	//	PPOINTF vpt(m_WorkingSplineLine.vx(m_nSelSplinePoint), m_WorkingSplineLine.vy(m_nSelSplinePoint));
	//	ChangeViewPoint(vpt);
	//	SetCursorPosSelectedPoint();

	//	return true;
	//} 
	else if (nChar == VK_DELETE) { //',/<'		
		if (m_bCheckOcclusion) {
			if (m_nSplineMouseOver != -1 && m_nOccMouseOver != -1) {
				m_RoadLane.line_ptr(m_nSplineMouseOver)->info.occlusions_top_bottom_.erase(
					m_RoadLane.line_ptr(m_nSplineMouseOver)->info.occlusions_top_bottom_.begin() + m_nOccMouseOver);
				m_nOccMouseOver = -1;
			}
		}
		else if (m_nSelSplinePoint > -1 && m_WorkingSplineLine.element_size() > m_nSelSplinePoint) {
			int new_idx = FindNextSplinePointIdx();
			if (new_idx > m_nSelSplinePoint) new_idx--;
			m_WorkingSplineLine.erase_element(m_nSelSplinePoint);
			m_nSelSplinePoint = -1;
			if (new_idx == -1) new_idx = FindBottomSplinePointIdx();
			m_nSelSplinePoint = new_idx;
			if (m_nSelSplinePoint > -1 && m_WorkingSplineLine.element_size() > m_nSelSplinePoint) {
				PPOINTF vpt(m_WorkingSplineLine.vx(m_nSelSplinePoint), m_WorkingSplineLine.vy(m_nSelSplinePoint));
				CRect rt;
				GetClientRect(rt);
				if (rt.PtInRect(vpt) == false) ChangeViewPoint(vpt);
				SetCursorPosSelectedPoint();
			}
			else {
				m_bControlSplinePoint = false;
				m_WorkingSplineLine.Reset();
				m_nSelSpline = -1;
			}
			return true;
		}
		else if (m_nSplineMouseOver != -1) {
			m_RoadLane.RemoveLine(m_nSplineMouseOver);
			m_nSplineMouseOver = -1;
			m_nSelSpline = -1;
			m_nSelSplinePoint = -1;
			return true;
		}
	}
	else if (nChar == 192) { //',/<', '`', '~'
		int idx = FindPrevSplinePointIdx();
		if (idx > -1) {
			m_nSelSplinePoint = idx;
			PPOINTF vpt(m_WorkingSplineLine.vx(m_nSelSplinePoint), m_WorkingSplineLine.vy(m_nSelSplinePoint));
			CRect rt;
			GetClientRect(rt);
			if (rt.PtInRect(vpt) == false) ChangeViewPoint(vpt);
			SetCursorPosSelectedPoint();
			return true;
		}
	}
	else if (nChar == 9) { //'./>' 'tab'
		int idx = FindNextSplinePointIdx();
		if (idx > -1) {
			m_nSelSplinePoint = idx;

			PPOINTF vpt(m_WorkingSplineLine.vx(m_nSelSplinePoint), m_WorkingSplineLine.vy(m_nSelSplinePoint));
			CRect rt;
			GetClientRect(rt);
			if (rt.PtInRect(vpt) == false) ChangeViewPoint(vpt);

			SetCursorPosSelectedPoint();

			return true;
		}
	}
	else if (nChar == VK_HOME) { //',/<'
		int idx = FindTopSplinePointIdx();
		if (idx > -1) {
			m_nSelSplinePoint = idx;
			PPOINTF vpt(m_WorkingSplineLine.vx(m_nSelSplinePoint), m_WorkingSplineLine.vy(m_nSelSplinePoint));
			CRect rt;
			GetClientRect(rt);
			if (rt.PtInRect(vpt) == false) ChangeViewPoint(vpt);
			SetCursorPosSelectedPoint();

			return true;
		}
	}
	else if (nChar == VK_END) { //'./>'
		int idx = FindBottomSplinePointIdx();
		if (idx > -1) {
			m_nSelSplinePoint = idx;
			PPOINTF vpt(m_WorkingSplineLine.vx(m_nSelSplinePoint), m_WorkingSplineLine.vy(m_nSelSplinePoint));
			CRect rt;
			GetClientRect(rt);
			if (rt.PtInRect(vpt) == false) ChangeViewPoint(vpt);
			SetCursorPosSelectedPoint();

			return true;
		}
	}
	else if (nChar == VK_RETURN) // Enter
	{
		if (m_WorkingSplineLine.initialized() == false) {
			if (m_WorkingSplineLine.element_size() == 2) {
				PPOINT3F middle_point(
					(m_WorkingSplineLine.spline_x_[0] + m_WorkingSplineLine.spline_x_[1]) * 0.5,
					(m_WorkingSplineLine.spline_y_[0] + m_WorkingSplineLine.spline_y_[1]) * 0.5,
					(m_WorkingSplineLine.line_r_[0] + m_WorkingSplineLine.line_r_[1]) * 0.5);
				m_WorkingSplineLine.AddPoint3(middle_point);
				m_WorkingSplineLine.GenerateModels();
			}
		}
		if (m_WorkingSplineLine.initialized()) {
			m_nSelSplinePoint = -1;

			LaneLine line = m_WorkingSplineLine;
			if (m_nSelSpline != -1) {
				RefineOCCRegion(line);
				m_RoadLane.ResetLine(line, m_nSelSpline);
				m_nSelSpline = -1;
			}
			else {
				m_RoadLane.AddLine(line);
			}
			m_WorkingSplineLine.Reset();
		}

		m_bControlSplinePoint = false;
		return true;
	}
	else if (nChar == _T('t') || nChar == _T('T')) // Enter
	{
		if (m_WorkingSplineLine.initialized() == false) {
			if (m_WorkingSplineLine.element_size() == 2) {
				PPOINT3F middle_point(
					(m_WorkingSplineLine.spline_x_[0] + m_WorkingSplineLine.spline_x_[1]) * 0.5,
					(m_WorkingSplineLine.spline_y_[0] + m_WorkingSplineLine.spline_y_[1]) * 0.5,
					(m_WorkingSplineLine.line_r_[0] + m_WorkingSplineLine.line_r_[1]) * 0.5);
				m_WorkingSplineLine.AddPoint3(middle_point);
				m_WorkingSplineLine.GenerateModels();
			}
		}
		if (m_WorkingSplineLine.initialized()) {
			m_nSelSplinePoint = -1;

			LaneLine line = m_WorkingSplineLine;
			if (m_nSelSpline != -1) {
				RefineOCCRegion(line);
				m_RoadLane.ResetLine(line, m_nSelSpline);
				m_nSelSpline = -1;
			}
			else {
				m_RoadLane.AddLine(line);
			}
			m_WorkingSplineLine.Reset();
		}

		m_bControlSplinePoint = false;
		return true;
	}
	//else if(nChar == VK_SPACE)
	//{
	//	if (m_WorkingSplineLine.element_size() == 0) {
	//		m_WorkingSplineLine.Reset();
	//		m_nSelSplinePoint = -1;
	//		m_nSelSpline = -1;
	//		POINT point;
	//		GetCursorPos(&point);
	//		::ScreenToClient(this->m_hWnd, &point);  // 마우스 커서 위치 좌표
	//		int sel_id = MousePointOnSpline(point);  // spline 위에 있는 지 확인
	//		if (sel_id != -1) {
	//			CSplineManagerDlg dlg;				 // type 분류 창 열기
	//			dlg.m_spline = m_RoadLane.line_ptr(sel_id);
	//			CPoint pos = point;
	//			ClientToScreen(&pos);
	//			dlg.window_pos = pos;
	//			CRect rect;
	//			GetClientRect(rect);
	//			dlg.left_oriented = rect.Width() * 0.5 < point.x ? false : true;
	//			dlg.DoModal();
	//		}
	//		else {
	//			m_bControlSplinePoint = false;
	//			m_bLbuttonControlPoint = false;
	//			POINT p;
	//			SetCursorPosSelectedPoint();
	//			GetCursorPos(&p);
	//			::ScreenToClient(this->m_hWnd, &p);
	//			ChangeViewPoint(p);
	//			SetCursorPosSelectedPoint();
	//		}
	//	}
	//	else {
	//		m_bControlSplinePoint = false;
	//		m_bLbuttonControlPoint = false;
	//		POINT p;
	//		SetCursorPosSelectedPoint();
	//		GetCursorPos(&p);
	//		::ScreenToClient(this->m_hWnd, &p);
	//		ChangeViewPoint(p);
	//		SetCursorPosSelectedPoint();
	//	}
	//	/*
	//	static bool zoom_in = false;
	//	static double prev_scale = g_view_scale;
	//	GetCursorPos(&p);
	//	::ScreenToClient(this->m_hWnd, &p);
	//	CRect rtClient;
	//	GetClientRect(&rtClient);
	//	if(zoom_in == false){
	//		zoom_in = true;
	//		prev_scale = g_view_scale;
	//		g_view_scale = 10 * (1/1.2);
	//		ZoomIn(CPoint(rtClient.Width() / 2, rtClient.Height() / 2));
	//	} else {
	//		zoom_in = false;
	//		g_view_scale = prev_scale * 1.2;
	//		ZoomOut(CPoint(rtClient.Width() / 2, rtClient.Height() / 2));
	//	}*/
	//	return true;
	//}
	else if (nChar == _T('6'))  // 차선 삭제
	{
		if (m_bCheckOcclusion) {
			if (m_nSplineMouseOver != -1 && m_nOccMouseOver != -1) {
				m_RoadLane.line_ptr(m_nSplineMouseOver)->info.occlusions_top_bottom_.erase(
					m_RoadLane.line_ptr(m_nSplineMouseOver)->info.occlusions_top_bottom_.begin() + m_nOccMouseOver);
				m_nOccMouseOver = -1;
			}
		}
		else if (m_nSelSplinePoint > -1 && m_WorkingSplineLine.element_size() > m_nSelSplinePoint) {
			int new_idx = FindNextSplinePointIdx();
			if (new_idx > m_nSelSplinePoint) new_idx--;
			m_WorkingSplineLine.erase_element(m_nSelSplinePoint);
			m_nSelSplinePoint = -1;
			if (new_idx == -1) new_idx = FindBottomSplinePointIdx();
			m_nSelSplinePoint = new_idx;
			if (m_nSelSplinePoint > -1 && m_WorkingSplineLine.element_size() > m_nSelSplinePoint) {
				PPOINTF vpt(m_WorkingSplineLine.vx(m_nSelSplinePoint), m_WorkingSplineLine.vy(m_nSelSplinePoint));
				CRect rt;
				GetClientRect(rt);
				if (rt.PtInRect(vpt) == false) ChangeViewPoint(vpt);
				SetCursorPosSelectedPoint();
			}
			else {
				m_bControlSplinePoint = false;
				m_WorkingSplineLine.Reset();
				m_nSelSpline = -1;
			}
			return true;
		}
		else if (m_nSplineMouseOver != -1) {
			m_RoadLane.RemoveLine(m_nSplineMouseOver);
			m_nSplineMouseOver = -1;
			m_nSelSpline = -1;
			m_nSelSplinePoint = -1;
			return true;
		}
	}

	return false;
}

void CPointingToolView::DrawWorkingSpline(
	Graphics &G, const CPoint &point,
	const Pen &pen_spline,
	const Pen &pen_spline_r,
	const Pen &pen_spline_r_expect,
	const Pen &pen_marker,
	const Pen &pen_marker_select,
	const Pen &pen_marker_expect)
{
	if (m_bLDownOcclusion) {
		G.DrawLine(&pen_marker_select, mapi2v_x(occStart.x - occStart.r - occStart.r), mapi2v_y(occStart.y), mapi2v_x(occStart.x + occStart.r + occStart.r), mapi2v_y(occStart.y));
		G.DrawLine(&pen_spline, mapi2v_x(occEnd.x - occEnd.r - occEnd.r), mapi2v_y(occEnd.y), mapi2v_x(occEnd.x + occEnd.r + occEnd.r), mapi2v_y(occEnd.y));

		Gdiplus::Rect rect_expect((int)(point.x - POINT_RECTANGLE_SIZE / 2), (int)point.y - POINT_RECTANGLE_SIZE / 2, POINT_RECTANGLE_SIZE, POINT_RECTANGLE_SIZE);
		rect_expect.X = point.x - 2;
		rect_expect.Y = point.y - 2;
		rect_expect.Width = 4;
		rect_expect.Height = 4;
		G.DrawEllipse(&pen_marker_select, rect_expect);
	}
	else if (m_WorkingSplineLine.element_size() >= 0) {
		vector<PointF> pts;
		int extension = 1;
		for (int i = 0; i < m_WorkingSplineLine.element_size(); ++i) {
			Gdiplus::RectF rect((m_WorkingSplineLine.vx(i) - POINT_RECTANGLE_SIZE / 2), m_WorkingSplineLine.vy(i) - POINT_RECTANGLE_SIZE / 2, POINT_RECTANGLE_SIZE, POINT_RECTANGLE_SIZE);
			if (rect.Contains(point.x, point.y))	extension = 0;
			pts.push_back(PointF(m_WorkingSplineLine.vx(i), m_WorkingSplineLine.vy(i)));
		}
		int mouse_pt_spoint_idx = -1;
		if (m_bControlSplinePoint) {
			mouse_pt_spoint_idx = MousePointOnSplinePoint(point);
			extension = mouse_pt_spoint_idx > -1 ? 0 : 1;
		}
		if (extension) {
			m_WorkingSplineLine.AddPoint3(PPOINT3F(mapv2i_x(point.x), mapv2i_y(point.y), m_WorkingSplineLine.reg_ir(mapv2i_y(point.y))));
			m_WorkingSplineLine.GenerateModels();
			pts.push_back(PointF(point.x, point.y));
		}

		if (m_WorkingSplineLine.initialized()) {
			pts.clear();
			double top_vy_ = mapi2v_y(m_WorkingSplineLine.top_y_);
			double bottom_vy_ = mapi2v_y(m_WorkingSplineLine.bottom_y_);
			for (int y = top_vy_; y <= bottom_vy_; y++) {
				double ix = m_WorkingSplineLine.reg_ix(mapv2i_y(y));
				pts.push_back(PPOINTF(mapi2v_x(ix), y));
			}
			if (m_bShowSplineOnly == false) {
				for (int i = 0; i < pts.size(); i++) {
					double r = m_WorkingSplineLine.reg_vr(mapv2i_y(pts[i].Y));
					double lx = pts[i].X - r;
					double rx = pts[i].X + r;
					G.DrawLine(&pen_spline_r_expect, lx, pts[i].Y, rx, pts[i].Y);
					if (i > 0) {
						double r_prev = m_WorkingSplineLine.reg_vr(mapv2i_y(pts[i - 1].Y));
						double prev_lx = pts[i - 1].X - r_prev;
						double prev_rx = pts[i - 1].X + r_prev;
						double overlap_ratio = (std::min(prev_rx, rx) - std::max(prev_lx, lx)) /
							(std::max(prev_rx, rx) - std::min(prev_lx, lx));
						if (overlap_ratio <= 0) {
							if (prev_lx > rx) {
								G.DrawLine(&pen_spline_r_expect, prev_lx, pts[i - 1].Y, rx, pts[i - 1].Y);
							}
							else {
								G.DrawLine(&pen_spline_r_expect, prev_rx, pts[i - 1].Y, lx, pts[i - 1].Y);
							}
						}
					}
				}
			}
		}

		G.DrawLines(&pen_spline, pts.data(), pts.size());

		Color clr_point = Color(std::min(g_pToolView->GetOverlayRatio() + 30, 255), 0, 0, 255);
		Pen pen_point(clr_point, 1);

		for (int i = 0; i < m_WorkingSplineLine.element_size() - extension; ++i) {
			Gdiplus::RectF rect((m_WorkingSplineLine.vx(i) - POINT_RECTANGLE_SIZE / 2), m_WorkingSplineLine.vy(i) - POINT_RECTANGLE_SIZE / 2, POINT_RECTANGLE_SIZE, POINT_RECTANGLE_SIZE);

			if (m_bControlSplinePoint == true && m_nSelSplinePoint == i)
				G.DrawLine(&pen_marker_select, std::lroundl(m_WorkingSplineLine.vx(i) - m_WorkingSplineLine.vr(i)), std::lroundl(m_WorkingSplineLine.vy(i)), std::lroundl(m_WorkingSplineLine.vx(i) + m_WorkingSplineLine.vr(i)), std::lround(m_WorkingSplineLine.vy(i)));
			else {
				G.DrawLine(&pen_spline_r, std::lroundl(m_WorkingSplineLine.vx(i) - m_WorkingSplineLine.vr(i)), std::lroundl(m_WorkingSplineLine.vy(i)), std::lroundl(m_WorkingSplineLine.vx(i) + m_WorkingSplineLine.vr(i)), std::lround(m_WorkingSplineLine.vy(i)));
				G.DrawRectangle(&pen_point, rect);
			}
		}
		//TRACE("%d\n", m_bOccControlPoint);
		if (m_bControlSplinePoint == false && m_bOccControlPoint == false) {
			//TRACE("tt\n");			
			if (extension == true && mouse_pt_spoint_idx == -1) {
				int idx = m_WorkingSplineLine.element_size() - extension;
				G.DrawLine(&pen_marker_expect, std::lroundl(m_WorkingSplineLine.vx(idx) - m_WorkingSplineLine.vr(idx)), std::lroundl(m_WorkingSplineLine.vy(idx)), std::lroundl(m_WorkingSplineLine.vx(idx) + m_WorkingSplineLine.vr(idx)), std::lround(m_WorkingSplineLine.vy(idx)));
			}
			Gdiplus::RectF rect_expect((int)(point.x - POINT_RECTANGLE_SIZE / 2), (int)point.y - POINT_RECTANGLE_SIZE / 2, POINT_RECTANGLE_SIZE, POINT_RECTANGLE_SIZE);
			rect_expect.X = point.x - 2;
			rect_expect.Y = point.y - 2;
			rect_expect.Width = 4;
			rect_expect.Height = 4;
			G.DrawEllipse(&pen_marker_select, rect_expect);
		}

		if (m_bControlSplinePoint == false && m_nSelSplinePoint >= 0 && m_nSelSplinePoint < m_WorkingSplineLine.element_size()) {
			Gdiplus::RectF rect((m_WorkingSplineLine.vx(m_nSelSplinePoint) - POINT_HIGHRIGHT_RECTANGLE_SIZE / 2), m_WorkingSplineLine.vy(m_nSelSplinePoint) - POINT_HIGHRIGHT_RECTANGLE_SIZE / 2, POINT_HIGHRIGHT_RECTANGLE_SIZE, POINT_HIGHRIGHT_RECTANGLE_SIZE);
			G.DrawRectangle(&pen_marker_select, rect);
		}

		if (extension) {
			m_WorkingSplineLine.erase_last_one();
		}
	}
	if (m_WorkingSplineLine.element_size() <= 0) {
		Pen pen_mouse_pt(Color(255, 0, 0), 2);
		Gdiplus::RectF rect_expect((int)(point.x - POINT_RECTANGLE_SIZE / 2), (int)point.y - POINT_RECTANGLE_SIZE / 2, POINT_RECTANGLE_SIZE, POINT_RECTANGLE_SIZE);
		rect_expect.X = point.x - 2;
		rect_expect.Y = point.y - 2;
		rect_expect.Width = 4;
		rect_expect.Height = 4;
		G.DrawEllipse(&pen_mouse_pt, rect_expect);
	}
}


void CPointingToolView::DrawSpline(
	Graphics &G,
	LaneLine &line,
	const Pen &pen_spline,
	const Pen &pen_spline_r,
	const Pen &pen_spline_r_expect,
	const Pen &pen_marker,
	const Pen &pen_occlusion,
	const SolidBrush &brush_text)
{
	vector<PointF> pts;
	vector<bool> occlusion;
	svld::tk::spline spline;
	std::vector<PPOINTF> xy;
	std::vector<PPOINTF> wy;

	double top_vy_ = mapi2v_y(line.top_y_);
	double bottom_vy_ = mapi2v_y(line.bottom_y_);
	std::vector<std::pair<float, float>> &occ = line.info.occlusions_top_bottom_;

	for (int y = top_vy_; y <= bottom_vy_; y++) {
		double ix = line.spline_xy_model_(mapv2i_y(y));
		pts.push_back(PPOINTF(mapi2v_x(ix), y));
		occlusion.push_back(false);
	}

	if (pts.size() == 0) return;

	for (int i = 0; i < occ.size(); i++) {
		int occ_top_vy_ = std::lroundl(mapi2v_y(occ[i].first));
		int occ_bottom_vy_ = std::lroundl(mapi2v_y(occ[i].second));

		for (int y = occ_top_vy_; y <= occ_bottom_vy_; y++) {
			int idx = y - top_vy_;
			if (idx > 0 && idx < occlusion.size())
				occlusion[idx] = true;
		}
	}

	//int I_R = std::lroundl(g_view_scale * m_pImage->GetHeight() / 2);
	//int center_x = std::lroundl(mapi2v_x(m_pImage->GetWidth()*0.5));
	//int center_y = std::lroundl(mapi2v_y(m_pImage->GetHeight()*0.5));
	//Pen guaird_line(Color(80, 255, 0, 0), 2);
	//G.DrawLine(&guaird_line, center_x, center_y - I_R, center_x, center_y + I_R);


	if (m_bShowSplineOnly == false) {
		for (int i = 0; i < pts.size(); i++) {
			double r = mapi2v_r(line.spline_ry_model_(mapv2i_y(pts[i].Y)));
			double lx = pts[i].X - r;
			double rx = pts[i].X + r;
			if (m_bCheckOcclusion && occlusion[i]) {
				G.DrawLine(&pen_occlusion, lx, pts[i].Y, rx, pts[i].Y);
			}
			else {
				G.DrawLine(&pen_spline_r_expect, lx, pts[i].Y, rx, pts[i].Y);
			}

			if (i > 0) {
				double r_prev = mapi2v_r(line.spline_ry_model_(mapv2i_y(pts[i - 1].Y)));
				double prev_lx = pts[i - 1].X - r_prev;
				double prev_rx = pts[i - 1].X + r_prev;
				double overlap_ratio = (std::min(prev_rx, rx) - std::max(prev_lx, lx)) /
					(std::max(prev_rx, rx) - std::min(prev_lx, lx));
				if (overlap_ratio <= 0) {
					if (prev_lx > rx) {
						if (m_bCheckOcclusion && occlusion[i]) {
							G.DrawLine(&pen_occlusion, prev_lx, pts[i - 1].Y, rx, pts[i - 1].Y);
						}
						else {
							G.DrawLine(&pen_spline_r_expect, prev_lx, pts[i - 1].Y, rx, pts[i - 1].Y);
						}
					}
					else {
						if (m_bCheckOcclusion && occlusion[i]) {
							G.DrawLine(&pen_occlusion, prev_rx, pts[i - 1].Y, lx, pts[i - 1].Y);
						}
						else {
							G.DrawLine(&pen_spline_r_expect, prev_rx, pts[i - 1].Y, lx, pts[i - 1].Y);
						}
					}
				}
			}
		}
	}

	if (!m_bCheckOcclusion) {
		G.DrawLines(&pen_spline, pts.data(), pts.size());
		for (int i = 0; i < line.spline_x_.size(); ++i) {
			/*Gdiplus::Rect rect(mapi2v_x(line.spline_x_[i]) - POINT_RECTANGLE_SIZE / 2, mapi2v_y(line.spline_y_[i]) - POINT_RECTANGLE_SIZE / 2, POINT_RECTANGLE_SIZE, POINT_RECTANGLE_SIZE);
			G.DrawRectangle(&pen_marker, rect);*/
			G.DrawLine(&pen_spline_r, mapi2v_x(line.spline_x_[i]) - mapi2v_r(line.line_r_[i]), mapi2v_y(line.spline_y_[i]), mapi2v_x(line.spline_x_[i]) + mapi2v_r(line.line_r_[i]), mapi2v_y(line.spline_y_[i]));
		}
	}

	PointF pt = pts[pts.size() - 1];
	RectF strRect(pt.X - 35, pt.Y, 70, 30);
	Gdiplus::Font font(_T("Arial"), 12, FontStyleBold, UnitPixel);

	StringFormat fmt;
	fmt.SetAlignment(StringAlignmentCenter);
	fmt.SetLineAlignment(StringAlignmentNear);

	CString str;
	WCHAR wstr[256];
	MultiByteToWideChar(0, 0, line.info.GetInfoText(), 256, wstr, 256);

	if (line.info.isEmpty()) {
		SolidBrush brush(Color(0, 0, 255));
		G.DrawString(wstr, -1, &font, strRect, &fmt, &brush);
	}
	else {
		G.DrawString(wstr, -1, &font, strRect, &fmt, &brush_text);
	}
}

void CPointingToolView::DrawSplines(const CPoint &point) {
	Color label = g_pToolView->GetLabelColor(), clr_spline;
	clr_spline = Color(g_pToolView->GetOverlayRatio(), label.GetR(), label.GetG(), label.GetB());
	Pen pen_spline(clr_spline, 1);
	Pen pen_spline_highlight(Color(g_pToolView->GetOverlayRatio(), 255, 0, 0), 3);

	Color clr_marker = Color(g_pToolView->GetOverlayRatio(), 0, 0, 0);
	Pen pen_marker(clr_marker, 1);

	Color clr_marker_expect = Color(std::max(g_pToolView->GetOverlayRatio() - 50, 0), 0, 0, 0);
	Pen pen_marker_expect(clr_marker_expect, 1);

	Color clr_marker_select = Color(std::min(g_pToolView->GetOverlayRatio() + 30, 255), 255, 0, 0);
	Pen pen_marker_select(clr_marker_select, 1);

	Color clr_spline_r = Color(std::min(g_pToolView->GetOverlayRatio() + 30, 255), 0, 0, 0);
	Pen pen_spline_r(clr_spline_r, 1);

	Color clr_occ = Color(std::min(g_pToolView->GetOverlayRatio() + 30, 255), 0, 0, 0);
	Pen pen_occ(clr_occ, 1);

	Color clr_spline_r_expect = Color(std::max(g_pToolView->GetOverlayRatio() - 60, 0), 0, 255, 50);  //// 스플라인만보이기 차선 색 ////
	Pen pen_spline_r_expect(clr_spline_r_expect, 1);

	Color clr_spline_r_expect_highlight = Color(std::max(g_pToolView->GetOverlayRatio(), 0), 0, 255, 50); //// 스플라인만보이기 차선 색 ////
	Pen pen_spline_r_expect_highlight(clr_spline_r_expect_highlight, 1);

	Color clr_vp = Color(std::max(g_pToolView->GetOverlayRatio(), 0), 255, 0, 0);
	Pen pen_vp(clr_vp, 3);
	Color clr_vp_highlight = Color(std::min(g_pToolView->GetOverlayRatio() + 60, 255), 255, 0, 0);
	Pen pen_vp_highlight(clr_vp_highlight, 5);

	RectF rect(0.0f, 0.0f, m_pViewCanvas->GetWidth(), m_pViewCanvas->GetHeight());
	Bitmap *temp = m_pViewCanvas->Clone(rect, PixelFormatDontCare);
	Graphics G(temp);
	G.SetSmoothingMode(SmoothingModeNone);

	//Spline 그릴때
	for (int i = 0; i < m_RoadLane.GetSizeLaneLine(); i++) {
		if (m_nSelSpline == -1 && m_WorkingSplineLine.element_size() == 0 && m_nSplineMouseOver == i) {

		}
		else if (m_nSelSpline != i) {
			DrawSpline(G, *m_RoadLane.line_ptr(i), pen_spline, pen_spline_r, pen_spline_r_expect, pen_marker, pen_occ, SolidBrush(Color(255, 0, 0)));
		}
	}
	// 마우스 올렸을 때
	for (int i = 0; i < m_RoadLane.GetSizeLaneLine(); i++) {
		if (m_nSelSpline == -1 && m_WorkingSplineLine.element_size() == 0 && m_nSplineMouseOver == i) {
			DrawSpline(G, *m_RoadLane.line_ptr(i), pen_spline_highlight, pen_spline_r, clr_spline_r_expect_highlight, pen_marker, pen_occ, SolidBrush(Color(0, 0, 255)));
		}
	}


	DrawWorkingSpline(G, point, pen_spline, pen_spline_r, pen_spline_r_expect, pen_marker, pen_marker_select, pen_marker_expect);

	if (g_pMainView->m_bVisibleVP == true && m_RoadLane.has_vp()) {
		float ivp_y = m_RoadLane.vp_y_ratio() * m_pImage->GetHeight();
		float ivp_x = m_RoadLane.vp_x_ratio() * m_pImage->GetWidth();
		int vvp_y = std::lroundl(mapi2v_y(ivp_y));
		int vvp_x = std::lroundl(mapi2v_x(ivp_x));
		CRect rt;
		GetClientRect(&rt);
		int r_x = std::lroundl(rt.Width() * 0.04);
		int r_y = std::lroundl(rt.Height() * 0.04);
		G.DrawLine(&pen_vp, vvp_x, vvp_y - r_y, vvp_x, vvp_y + r_y);
		G.DrawLine(&pen_vp, vvp_x - r_x, vvp_y, vvp_x + r_x, vvp_y);
		if (m_bMouseOnVP) {
			if (m_bVerticalVP) {
				G.DrawLine(&pen_vp_highlight, vvp_x, vvp_y - r_y, vvp_x, vvp_y + r_y);

			}
			else {
				G.DrawLine(&pen_vp_highlight, vvp_x - r_x, vvp_y, vvp_x + r_x, vvp_y);

			}
		}
	}

	CDC *pDC = GetDC();
	Graphics G2(pDC->m_hDC);
	if (m_pCachedViewCanvas) ::delete m_pCachedViewCanvas;
	m_pCachedViewCanvas = ::new CachedBitmap(temp, &G2);
	ReleaseDC(pDC);
	delete temp;
	Invalidate(FALSE);
	UpdateWindow();
}

void CPointingToolView::DrawSplines(Graphics &G, const CPoint &point)
{
	Color label = g_pToolView->GetLabelColor(0), clr_spline;
	clr_spline = Color(g_pToolView->GetOverlayRatio(), label.GetR(), label.GetG(), label.GetB());
	Pen pen_spline(clr_spline, 1);
	Pen pen_spline_highlight(Color(g_pToolView->GetOverlayRatio(), 255, 0, 0), 3);

	Color clr_marker = Color(g_pToolView->GetOverlayRatio(), 0, 0, 0);
	Pen pen_marker(clr_marker, 1);

	Color clr_marker_expect = Color(std::max(g_pToolView->GetOverlayRatio() - 50, 0), 0, 0, 0);
	Pen pen_marker_expect(clr_marker_expect, 1);

	Color clr_marker_select = Color(std::min(g_pToolView->GetOverlayRatio() + 30, 255), 255, 0, 0);
	Pen pen_marker_select(clr_marker_select, 1);

	Color clr_spline_r = Color(std::min(g_pToolView->GetOverlayRatio() + 30, 255), 0, 0, 0);
	Pen pen_spline_r(clr_spline_r, 1);

	Color clr_occ = Color(std::min(g_pToolView->GetOverlayRatio() + 30, 255), 0, 0, 0);
	Pen pen_occ(clr_occ, 1);

	Color clr_spline_r_expect = Color(std::max(g_pToolView->GetOverlayRatio() - 60, 0), 0, 255, 50);  //// 스플라인만보이기 차선 색 ////
	Pen pen_spline_r_expect(clr_spline_r_expect, 1);

	Color clr_spline_r_expect_highlight = Color(std::max(g_pToolView->GetOverlayRatio(), 0), 0, 255, 50); //// 스플라인만보이기 차선 색 ////
	Pen pen_spline_r_expect_highlight(clr_spline_r_expect_highlight, 1);

	RectF rect(0.0f, 0.0f, m_pViewCanvas->GetWidth(), m_pViewCanvas->GetHeight());

	for (int i = 0; i < m_RoadLane.GetSizeLaneLine(); i++) {
		if (m_nSelSpline == -1 && m_WorkingSplineLine.element_size() == 0 && m_nSplineMouseOver == i) {

		}
		else if (m_nSelSpline != i) {
			DrawSpline(G, *m_RoadLane.line_ptr(i), pen_spline, pen_spline_r, pen_spline_r_expect, pen_marker, pen_occ, SolidBrush(Color(255, 0, 0)));
		}
	}
	for (int i = 0; i < m_RoadLane.GetSizeLaneLine(); i++) {
		if (m_nSelSpline == -1 && m_WorkingSplineLine.element_size() == 0 && m_nSplineMouseOver == i) {
			DrawSpline(G, *m_RoadLane.line_ptr(i), pen_spline_highlight, pen_spline_r, clr_spline_r_expect_highlight, pen_marker, pen_occ, SolidBrush(Color(0, 0, 255)));
		}
	}


	DrawWorkingSpline(G, point, pen_spline, pen_spline_r, pen_spline_r_expect, pen_marker, pen_marker_select, pen_marker_expect);
}

int CPointingToolView::FindNextSplinePointIdx()
{
	int min_below_idx = -1;
	if (m_nSelSplinePoint >= 0 && m_nSelSplinePoint < m_WorkingSplineLine.element_size()) {
		float min_dist = 10000000000.f;
		for (int i = 0; i < m_WorkingSplineLine.element_size(); i++) {
			if (i == m_nSelSplinePoint) continue;
			if (m_WorkingSplineLine.iy(i) <= m_WorkingSplineLine.iy(m_nSelSplinePoint)) continue;
			float dist = m_WorkingSplineLine.iy(i) - m_WorkingSplineLine.iy(m_nSelSplinePoint);
			if (dist < min_dist) {
				min_below_idx = i;
				min_dist = dist;
			}
		}
	}
	else {
		return FindTopSplinePointIdx();
	}
	return min_below_idx;
}

int CPointingToolView::FindPrevSplinePointIdx()
{
	int min_upper_idx = -1;
	if (m_nSelSplinePoint >= 0 && m_nSelSplinePoint < m_WorkingSplineLine.element_size()) {
		float min_dist = 10000000000.f;
		for (int i = 0; i < m_WorkingSplineLine.element_size(); i++) {
			if (i == m_nSelSplinePoint) continue;
			if (m_WorkingSplineLine.iy(i) >= m_WorkingSplineLine.iy(m_nSelSplinePoint)) continue;
			float dist = m_WorkingSplineLine.iy(m_nSelSplinePoint) - m_WorkingSplineLine.iy(i);
			if (dist < min_dist) {
				min_upper_idx = i;
				min_dist = dist;
			}
		}
	}
	return min_upper_idx;
}

int CPointingToolView::FindTopSplinePointIdx()
{
	int top_idx = -1;
	float max_top_y = 10000000000000.f;
	for (int i = 0; i < m_WorkingSplineLine.element_size(); i++) {
		if (i == m_nSelSplinePoint) continue;
		if (max_top_y > m_WorkingSplineLine.iy(i)) {
			max_top_y = m_WorkingSplineLine.iy(i);
			top_idx = i;
		}
	}
	return top_idx;
}

int CPointingToolView::FindBottomSplinePointIdx()
{
	int bottom_idx = -1;

	float max_bottom_y = -1;
	for (int i = 0; i < m_WorkingSplineLine.element_size(); i++) {
		if (i == m_nSelSplinePoint) continue;
		if (max_bottom_y < m_WorkingSplineLine.iy(i)) {
			max_bottom_y = m_WorkingSplineLine.iy(i);
			bottom_idx = i;
		}
	}
	return bottom_idx;
}

int CPointingToolView::MousePointOnSpline(const CPoint &point)
{
	if (m_WorkingSplineLine.element_size()) return -1;
	PPOINTF view_pt(point.x, point.y);
	PPOINTF image_pt = mapv2i(view_pt);
	for (int i = 0; i < m_RoadLane.GetSizeLaneLine(); i++) {
		LaneLine &line = *m_RoadLane.line_ptr(i);
		if (line.top_y_ > image_pt.y || line.bottom_y_ < image_pt.y) continue;
		PPOINT3F point3 = line.EstimatePoint(image_pt.y);
		if (fabs(point3.x - image_pt.x) < (point3.r + 1)) {
			return i;
		}
	}
	return -1;
}

int CPointingToolView::MousePointOnSplinePoint(const CPoint &point)
{
	if (m_WorkingSplineLine.element_size() == 0) return -1;
	float radius = std::ceil(POINT_HIGHRIGHT_RECTANGLE_SIZE / 2.f);
	//radius = std::min(radius, 1.f);
	for (int i = 0; i < m_WorkingSplineLine.element_size(); i++) {
		double x = m_WorkingSplineLine.vx(i);
		double y = m_WorkingSplineLine.vy(i);
		CRect rt(x - radius, y - radius, x + radius, y + radius);
		if (rt.PtInRect(point)) { return i; }
	}
	return -1;
}



bool CPointingToolView::MousePointOnVP(const CPoint &point)
{
	if (m_RoadLane.has_vp() == false) return false;
	if (m_bVisibleVP == false) return false;
	CRect rtVP_V, rtVP_H;
	CRect rtClient;
	GetClientRect(&rtClient);
	float ivp_y = m_RoadLane.vp_y_ratio() * m_pImage->GetHeight();
	float ivp_x = m_RoadLane.vp_x_ratio() * m_pImage->GetWidth();
	int vvp_y = std::lroundl(mapi2v_y(ivp_y));
	int vvp_x = std::lroundl(mapi2v_x(ivp_x));

	CRect rt;
	GetClientRect(&rt);
	int rw = std::lroundl(rt.Width() * 0.04);
	int rh = std::lroundl(rt.Height() * 0.04);

	rtVP_H.left = vvp_x - rw;
	rtVP_H.right = vvp_x + rw;
	rtVP_H.top = vvp_y - 3;
	rtVP_H.bottom = vvp_y + 3;


	rtVP_V.left = vvp_x - 3;
	rtVP_V.right = vvp_x + 3;
	rtVP_V.top = vvp_y - rh;
	rtVP_V.bottom = vvp_y + rh;

	if (rtVP_H.PtInRect(point)) {
		m_bVerticalVP = false;
	}
	else if (rtVP_V.PtInRect(point)) {
		m_bVerticalVP = true;
	}
	else {
		return false;
	}

	return true;
}

void CPointingToolView::LoadDataSpline()
{
	if (m_pImage) {
		CString fname = g_FileList[g_CurFileIdx].first;
		CString lane_info_fname = fname;
		lane_info_fname = lane_info_fname.Left(lane_info_fname.ReverseFind('.')) + _T(".xml"); /////// 수정 ///////
		TCHAR full_lane_path[256];
		_stprintf(full_lane_path, _T("%s/%s"), g_pToolView->m_strMaskFolder.GetBuffer(), lane_info_fname.GetBuffer());
		char ctemp[1024];
		WideCharToMultiByte(CP_ACP, 0, full_lane_path, 1024, ctemp, 1024, NULL, NULL);
		m_RoadLane.ReadFile(ctemp);
		m_BackupRoadLane = m_RoadLane;
	}
}
void CPointingToolView::SaveDataSpline()
{
	if (m_pImage) {
		CString fname = g_FileList[g_CurFileIdx].first;
		CString lane_info_fname = fname;
		lane_info_fname = lane_info_fname.Left(lane_info_fname.ReverseFind('.')) + _T(".xml"); ////// 수정 ///////
		TCHAR full_lane_path[1024];
		_stprintf(full_lane_path, _T("%s\\%s"), g_pToolView->m_strMaskFolder.GetBuffer(), lane_info_fname.GetBuffer());

		char ctemp[1024];
		WideCharToMultiByte(CP_ACP, 0, full_lane_path, 1024, ctemp, 1024, NULL, NULL);
		m_RoadLane.WriteFile(ctemp);
	}
}

void CPointingToolView::DrawSplineToMask(RoadLaneManager &lane, Graphics &G, int width, int height) {
	const int COLOR_ID_STEP = 10;
	int color_value = COLOR_ID_STEP;

	//G.Clear(Color(0, 0, 0));

	//if(lane.has_vp()){
	//	int vpy = std::lroundl(lane.vp_y_ratio() * height);

	//	Gdiplus::Rect rect_under_vp(0, vpy, width, height - vpy);
	//	Gdiplus::Rect rect_vp(0, 0, width, vpy);
	//	Color clr_under_vp = Color(127, 127, 127);
	//	Color clr_vp = Color(0, 0, 255);
	//	SolidBrush bru_under_vp(clr_under_vp);
	//	SolidBrush bru_pen_vp(clr_vp);

	//	G.FillRectangle(&bru_under_vp, rect_under_vp);
	//	G.FillRectangle(&bru_pen_vp, rect_vp);
	//}

	if (lane.has_vp()) {
		G.Clear(Color(127, 127, 127));
		int vpy = std::lroundl(lane.vp_y_ratio() * height);
		int vpx = std::lroundl(lane.vp_x_ratio() * width);
		int r_x = std::lroundl(width * 0.02);
		int r_y = std::lroundl(height * 0.02);

		//Gdiplus::Rect rect_under_vp(0, vpy, width, height - vpy);
		Gdiplus::Rect rect_vp(vpx - r_x, vpy - r_y * 2, r_x * 2, r_y * 2);
		Color clr_vp = Color(0, 0, 255);
		//SolidBrush bru_under_vp(clr_under_vp);
		SolidBrush bru_pen_vp(clr_vp);

		//G.FillRectangle(&bru_under_vp, rect_under_vp);
		G.FillRectangle(&bru_pen_vp, rect_vp);
	}

	for (int i = 0; i < lane.GetSizeLaneLine(); i++) {
		LaneLine &line = *lane.line_ptr(i);
		vector<PointF> pts;
		std::vector<PPOINTF> xy;
		std::vector<PPOINTF> wy;
		std::vector<bool> occlusions;
		for (int y = line.top_y_; y <= line.bottom_y_; y++) {
			bool occlusion = false;
			for (int k = 0; k < line.info.occlusions_top_bottom_.size(); k++) {
				if (y >= line.info.occlusions_top_bottom_[k].first &&
					y <= line.info.occlusions_top_bottom_[k].second) {
					occlusion = true;
					break;
				}
			}
			occlusions.push_back(occlusion);
			double ix = line.spline_xy_model_(y);
			pts.push_back(PPOINTF(ix, y));
		}

		Color clr_spline = MakeColor(color_value);
		Color clr_occlusion = Color(clr_spline.GetR(), clr_spline.GetB(), 127);
		Pen pen_spline_r(clr_spline, 1);
		Pen pen_spline_occ(clr_occlusion, 1);
		color_value += COLOR_ID_STEP;
		for (int i = 0; i < pts.size(); i++) {
			double r = line.spline_ry_model_(pts[i].Y);
			double lx = pts[i].X - r;
			double rx = pts[i].X + r;
			if (occlusions[i]) {
				G.DrawLine(&pen_spline_occ, lx, pts[i].Y, rx, pts[i].Y);
			}
			else {
				G.DrawLine(&pen_spline_r, lx, pts[i].Y, rx, pts[i].Y);
			}

			if (i > 0 && pts[i].Y - pts[i - 1].Y < 3) {
				double r_prev = line.spline_ry_model_(pts[i - 1].Y);
				double prev_lx = pts[i - 1].X - r_prev;
				double prev_rx = pts[i - 1].X + r_prev;
				double overlap_ratio = (std::min(prev_rx, rx) - std::max(prev_lx, lx)) /
					(std::max(prev_rx, rx) - std::min(prev_lx, lx));
				if (overlap_ratio <= 0) {
					if (prev_lx > rx) {
						if (occlusions[i]) {
							G.DrawLine(&pen_spline_occ, prev_lx, pts[i - 1].Y, rx, pts[i - 1].Y);
						}
						else {
							G.DrawLine(&pen_spline_r, prev_lx, pts[i - 1].Y, rx, pts[i - 1].Y);
						}

					}
					else {
						if (occlusions[i]) {
							G.DrawLine(&pen_spline_occ, prev_rx, pts[i - 1].Y, lx, pts[i - 1].Y);
						}
						else {
							G.DrawLine(&pen_spline_r, prev_rx, pts[i - 1].Y, lx, pts[i - 1].Y);
						}

					}
				}
			}
		}
	}
}

void CPointingToolView::SaveSplineMaskImage()
{
	if (m_pImage) {
		int w = m_pImage->GetWidth(), h = m_pImage->GetHeight();
		Bitmap mask_bmp(w, h, PixelFormat24bppRGB);
		RoadLaneManager road_lane = m_RoadLane;
		road_lane.SortingLength();
		DrawRoadMarkingToMask(road_lane, mask_bmp);
		DrawVPToMask(road_lane, mask_bmp);
		CString fname = g_FileList[g_CurFileIdx].first;
		char cfull_xml_path[256];
		TCHAR full_xml_path[256];
		TCHAR full_xml_folder_path[256];
		CString xml_fname = fname;
		xml_fname = xml_fname.Left(xml_fname.ReverseFind('.')) + _T(".xml");
		_stprintf(full_xml_folder_path, _T("%s\\TypeData"), g_pToolView->m_strMaskFolder.GetBuffer());
		CreateDirectory(full_xml_folder_path, NULL);
		_stprintf(full_xml_path, _T("%s\\TypeData\\%s"), g_pToolView->m_strMaskFolder.GetBuffer(), xml_fname.GetBuffer());
		WideCharToMultiByte(CP_ACP, 0, full_xml_path, 256, cfull_xml_path, 256, NULL, NULL);
		DrawLaneBoundaryToMask(road_lane, mask_bmp, cfull_xml_path);
		CString mask_fname = fname;
		mask_fname = mask_fname.Left(mask_fname.ReverseFind('.')) + _T(".png");
		TCHAR full_mask_path[256];
		TCHAR full_original_path[256];
		TCHAR full_folder_path[256];
		_stprintf(full_folder_path, _T("%s\\LaneData"), g_pToolView->m_strMaskFolder.GetBuffer());
		_stprintf(full_mask_path, _T("%s\\LaneData\\%s"), g_pToolView->m_strMaskFolder.GetBuffer(), mask_fname.GetBuffer());
		_stprintf(full_original_path, _T("%s\\LaneData\\%s"), g_pToolView->m_strMaskFolder.GetBuffer(), fname.GetBuffer());
		CreateDirectory(full_folder_path, NULL);
		CLSID clsid;
		GetEncCLSID(L"image/png", &clsid);
		mask_bmp.Save(full_mask_path, &clsid, NULL);
	}
}

void CPointingToolView::SaveSplineMaskImageAll()
{
	TCHAR full_img_path[1024];
	TCHAR full_mask_path[1024];
	TCHAR full_lane_path[1024];

	for (int i = 0; i < g_FileList.size(); i++) {
		CString fname = g_FileList[i].first;
		//CString mask_fname = fname;
		CString lane_fname;
		_stprintf(full_img_path, _T("%s/%s"), g_pToolView->m_strImageFolder.GetBuffer(), fname.GetBuffer());
		lane_fname = fname.Left(fname.ReverseFind('.')) + _T(".xml");
		_stprintf(full_lane_path, _T("%s/%s"), g_pToolView->m_strMaskFolder.GetBuffer(), lane_fname.GetBuffer());
		/*mask_fname = mask_fname.Left(mask_fname.ReverseFind('.')) + _T(".png");
		_stprintf(full_mask_path, _T("%s/%s"), g_pToolView->m_strMaskFolder.GetBuffer(), mask_fname.GetBuffer());		*/

		Bitmap *pImage = LoadImgFromFile(full_img_path);

		if (pImage) {
			char ctemp[1024];
			WideCharToMultiByte(CP_ACP, 0, full_lane_path, 1024, ctemp, 1024, NULL, NULL);
			RoadLaneManager road_lane;

			int w = pImage->GetWidth(), h = pImage->GetHeight();
			Bitmap mask_bmp(w, h, PixelFormat24bppRGB);
			Graphics G(&mask_bmp);
			G.Clear(Color(0, 0, 0));
			if (road_lane.ReadFile(ctemp)) {
				road_lane.SortingLength();
				DrawRoadMarkingToMask(road_lane, mask_bmp);
				DrawVPToMask(road_lane, mask_bmp);
				// make type.xml path
				char cfull_xml_path[256];
				TCHAR full_xml_path[256];
				TCHAR full_xml_folder_path[256];
				CString xml_fname = fname;
				xml_fname = xml_fname.Left(xml_fname.ReverseFind('.')) + _T(".xml");
				_stprintf(full_xml_folder_path, _T("%s\\TypeData"), g_pToolView->m_strMaskFolder.GetBuffer());
				CreateDirectory(full_xml_folder_path, NULL);
				_stprintf(full_xml_path, _T("%s\\TypeData\\%s"), g_pToolView->m_strMaskFolder.GetBuffer(), xml_fname.GetBuffer());
				WideCharToMultiByte(CP_ACP, 0, full_xml_path, 256, cfull_xml_path, 256, NULL, NULL);
				// draw lane boundary mask
				DrawLaneBoundaryToMask(road_lane, mask_bmp, cfull_xml_path);
				// save mask
				CString mask_fname = fname;
				mask_fname = mask_fname.Left(mask_fname.ReverseFind('.')) + _T(".png");
				TCHAR full_mask_path[256];
				TCHAR full_original_path[256];
				TCHAR full_folder_path[256];
				_stprintf(full_folder_path, _T("%s\\LaneData"), g_pToolView->m_strMaskFolder.GetBuffer());
				_stprintf(full_mask_path, _T("%s\\LaneData\\%s"), g_pToolView->m_strMaskFolder.GetBuffer(), mask_fname.GetBuffer());
				_stprintf(full_original_path, _T("%s\\LaneData\\%s"), g_pToolView->m_strMaskFolder.GetBuffer(), fname.GetBuffer());
				CreateDirectory(full_folder_path, NULL);
				CLSID clsid;
				GetEncCLSID(L"image/png", &clsid);
				mask_bmp.Save(full_mask_path, &clsid, NULL);
				//GetEncCLSID(L"image/jpg", &clsid);
				//pImage->Save(full_original_path, &clsid, NULL);

			}
			::delete pImage;
			pImage = NULL;
		}
	}
}

//Lane Mask Format
//Draw order: Roadmaker -> VP -> Lane
//R: occ(1bit), ext(1bit), LaneID(6bit)
//G: Roamaker(1bit), Shape(2bit), Pos(5bit)
//B: Color(2bit) + VP(0,64,128)(x,?,O)


void CPointingToolView::DrawRoadMarkingToMask(RoadLaneManager &road, Bitmap &mask)
{
	int w = mask.GetWidth(), h = mask.GetHeight();
	Rect rect(0, 0, w, h);

	Graphics G(&mask);
	SolidBrush br(Color(0, 128, 0));
	for (int i = 0; i < road.GetSizeRoadMarking(); i++) {
		const RoadMarkingPolygon &polygons = *road.roadmarking_ptr(i);
		if (polygons.GetPolygonPointNum())
		{
			std::vector<Gdiplus::PointF> points(polygons.GetPolygonPointNum());
			for (int j = 0; j < polygons.GetPolygonPointNum(); j++) {
				PPOINTF pointf = polygons.GetPoint(j);
				points[j].X = pointf.x;
				points[j].Y = pointf.y;
			}
			G.FillPolygon(&br, &points[0], points.size());
		}
	}
}

void CPointingToolView::DrawVPToMask(RoadLaneManager &road, Bitmap &mask)
{
	Rect rt_vp;
	int b;
	int w = mask.GetWidth(), h = mask.GetHeight();
	if (!road.has_vp()) {
		b = 64;
		rt_vp = Rect(0, 0, w, h);
	}
	else
	{
		b = 128;
		int rt_width = w / 5;
		int rt_height = h / 8;
		if (road.IsZF())
		{
			rt_width = w / 8;
			rt_height = h / 24;
		}

		int vpx = (int)std::round(road.vp_x_ratio() * w);
		int vpy = (int)std::round(road.vp_y_ratio() * h);

		rt_vp = Rect(vpx - rt_width / 2, vpy - rt_height / 2, rt_width, rt_height);
	}

	Rect rect(0, 0, w, h);

	Gdiplus::BitmapData bmData_o;
	mask.LockBits(&rect, Gdiplus::ImageLockModeWrite, PixelFormat24bppRGB, &bmData_o);
	for (int y = max(0, rt_vp.Y); y < rt_vp.Y + rt_vp.Height && y < h; y++)
	{
		for (int x = max(0, rt_vp.X); x < rt_vp.X + rt_vp.Width && x < w; x++)
		{
			BYTE *pbyte = (BYTE *)bmData_o.Scan0;
			pbyte[y*bmData_o.Stride + x * 3] = b;
		}
	}
	mask.UnlockBits(&bmData_o);
}

bool compareLaneType(LaneLine l0, LaneLine l1)
{
	LaneInfo info0 = l0.info;
	LaneInfo info1 = l1.info;

	if (info0.GetType3() == info1.GetType3() && info0.GetType4() == info1.GetType4() && info0.GetType3_ID() == info1.GetType3_ID())
	{
		if (info0.GetType2() == LaneInfo::DOUBLE)
			return true;
		return false;
	}

	if (info0.GetType3() == LaneInfo::UNCERTAIN)
	{
		return true;
	}
	else if (info1.GetType3() == LaneInfo::UNCERTAIN)
	{
		return false;
	}

	if (info0.GetType4() == LaneInfo::OPPOSITE_SIDE)
	{
		return true;
	}
	else if (info1.GetType4() == LaneInfo::OPPOSITE_SIDE)
	{
		return false;
	}

	if (info0.GetType4() == LaneInfo::BRANCH || info0.GetType4() == LaneInfo::MERGED)
	{
		return true;
	}
	else if (info1.GetType4() == LaneInfo::BRANCH || info1.GetType4() == LaneInfo::MERGED)
	{
		return false;
	}

	return true;
}

int CPointingToolView::DrawLaneToMask(RoadLaneManager &road, Bitmap &mask)
{
	static int max_id = 0;
	int sl_count = 0;
	int dl_count = 0;
	vector<LaneLine> lines;
	static int img_id = 0;
	img_id++;
	for (int i = 0; i < road.GetSizeLaneLine(); i++)
	{

		LaneInfo info = road.line_ptr(i)->info;
		if (LaneInfo::SOLID <= info.GetType1() && info.GetType1() <= LaneInfo::CATS_EYE && LaneInfo::SINGLE <= info.GetType2() && info.GetType2() <= LaneInfo::ACCESSORIE && LaneInfo::LEFT <= info.GetType3() && info.GetType3() <= LaneInfo::UNCERTAIN && LaneInfo::C4_NONE <= info.GetType4() && info.GetType4() <= LaneInfo::OPPOSITE_SIDE)
		{
			if (info.GetType4() != LaneInfo::UNABLE)
			{
				lines.push_back(*(road.line_ptr(i)));
			}
		}
	}

	for (int i = 0; i < lines.size(); i++)
	{
		int select_idx = i;
		for (int j = i + 1; j < lines.size(); j++)
		{
			if (compareLaneType(lines[j], lines[select_idx]))
				select_idx = j;
		}
		LaneLine temp = lines[i];
		lines[i] = lines[select_idx];
		lines[select_idx] = temp;
	}

	for (int i = 0; i < lines.size(); i++)
	{
		int id = -1;

		LaneInfo info = lines[i].info;
		if (info.GetType2() == LaneInfo::SINGLE)
			id = 1 + 2 * sl_count++;
		else
			id = 2 * (1 + dl_count++);

		DrawLineToMask(lines[i], id, mask);
	}
	return 1;
}

void CPointingToolView::DrawLineAsRoadMarkerToMask(LaneLine &line, Bitmap &mask) {
	int w = mask.GetWidth(), h = mask.GetHeight();
	Rect rect(0, 0, w, h);

	Gdiplus::BitmapData bmData_o;
	mask.LockBits(&rect, Gdiplus::ImageLockModeWrite, PixelFormat24bppRGB, &bmData_o);
	vector<PointF> pts;

	int top_y = std::max((int)std::round(line.top_y_), 0);
	int bottom_y = std::min((int)std::round(line.bottom_y_), h - 1);

	for (int y = top_y; y <= bottom_y; y++) {
		double ix = line.spline_xy_model_(y);
		pts.push_back(PPOINTF(ix, y));
	}
	int y = top_y;
	BYTE *pbyte = (BYTE *)bmData_o.Scan0 + y * bmData_o.Stride;
	int prev_lx, prev_rx;
	for (int i = 0; i < pts.size(); i++) {
		double r = line.spline_ry_model_(pts[i].Y);
		int lx = std::round(pts[i].X - r);
		int rx = std::round(pts[i].X + r);
		for (int j = lx; j <= rx; j++) {
			if (j < 0 || j > w - 1) continue;
			if (pbyte[j * 3 + 1] < 128)
				pbyte[j * 3 + 1] += 128;
		}
		if (i) {
			double overlap_ratio = (std::min(prev_lx, rx) - std::max(prev_lx, lx)) /
				(double)(std::max(prev_rx, rx) - std::min(prev_lx, lx));
			if (overlap_ratio <= 0) {
				if (prev_lx > rx) {
					for (int j = prev_lx; j <= rx; j++) {
						if (j < 0 || j > w - 1) continue;
						if (pbyte[j * 3 + 1] < 128)
							pbyte[j * 3 + 1] += 128;
					}
				}
				else {
					for (int j = prev_lx; j <= rx; j++) {
						if (j < 0 || j > w - 1) continue;
						if (pbyte[j * 3 + 1] < 128)
							pbyte[j * 3 + 1] += 128;
					}
				}
			}
		}
		prev_lx = lx;
		prev_rx = rx;
		pbyte += bmData_o.Stride;
	}
	mask.UnlockBits(&bmData_o);
}
void CPointingToolView::DrawLineToMask(LaneLine &line, int id, Bitmap &mask)
{
	int w = mask.GetWidth(), h = mask.GetHeight();
	Rect rect(0, 0, w, h);

	Gdiplus::BitmapData bmData_o;
	mask.LockBits(&rect, Gdiplus::ImageLockModeWrite, PixelFormat24bppRGB, &bmData_o);
	vector<PointF> pts;
	std::vector<PPOINTF> xy;
	std::vector<PPOINTF> wy;
	std::vector<bool> occlusions;

	int typePos = 0;
	int typeShape = 0;
	LaneInfo info = line.info;

	if (info.GetType4() == LaneInfo::OPPOSITE_SIDE && info.GetType3() == LaneInfo::LEFT)
		typePos = 0;
	else if (info.GetType4() == LaneInfo::BRANCH)
		typePos = 1;
	else if (info.GetType4() == LaneInfo::C4_NONE && info.GetType3() == LaneInfo::LEFT && info.GetType3_ID() == 3)
		typePos = 2;
	else if (info.GetType4() == LaneInfo::C4_NONE && info.GetType3() == LaneInfo::LEFT && info.GetType3_ID() == 2)
		typePos = 3;
	else if (info.GetType4() == LaneInfo::C4_NONE && info.GetType3() == LaneInfo::LEFT && info.GetType3_ID() == 1)
		typePos = 4;
	else if (info.GetType4() == LaneInfo::C4_NONE && info.GetType3() == LaneInfo::LEFT && info.GetType3_ID() == 0)
		typePos = 5;
	else if (info.GetType4() == LaneInfo::C4_NONE && info.GetType3() == LaneInfo::RIGHT && info.GetType3_ID() == 0)
		typePos = 6;
	else if (info.GetType4() == LaneInfo::C4_NONE && info.GetType3() == LaneInfo::RIGHT && info.GetType3_ID() == 1)
		typePos = 7;
	else if (info.GetType4() == LaneInfo::C4_NONE && info.GetType3() == LaneInfo::RIGHT && info.GetType3_ID() == 2)
		typePos = 8;
	else if (info.GetType4() == LaneInfo::C4_NONE && info.GetType3() == LaneInfo::RIGHT && info.GetType3_ID() == 3)
		typePos = 9;
	else if (info.GetType4() == LaneInfo::MERGED)
		typePos = 10;
	else if (info.GetType4() == LaneInfo::OPPOSITE_SIDE && info.GetType3() == LaneInfo::RIGHT)
		typePos = 11;
	else  //uncertain
		typePos = 12;

	typeShape = info.GetType1() - 1;

	int green = (typeShape << 5) + typePos;

	int blue = info.GetType5() - 1;


	int top_y = std::max((int)std::round(line.top_y_), 0);
	int bottom_y = std::min((int)std::round(line.bottom_y_), h - 1);

	bool b_ext = (4 <= typePos && typePos <= 7);
	for (int y = top_y; y <= (b_ext ? (h - 1) : bottom_y); y++) {
		bool occlusion = false;
		for (int k = 0; k < line.info.occlusions_top_bottom_.size(); k++) {
			if (y >= line.info.occlusions_top_bottom_[k].first &&
				y <= line.info.occlusions_top_bottom_[k].second) {
				occlusion = true;
				break;
			}
		}
		occlusions.push_back(occlusion);
		double ix = line.spline_xy_model_(y);
		pts.push_back(PPOINTF(ix, y));
	}
	int y = top_y;
	BYTE *pbyte = (BYTE *)bmData_o.Scan0 + y * bmData_o.Stride;
	int prev_lx, prev_rx;
	for (int i = 0; i < pts.size(); i++) {
		double r = line.spline_ry_model_(pts[i].Y);
		int lx = std::round(pts[i].X - r);
		int rx = std::round(pts[i].X + r);
		for (int j = lx; j <= rx; j++) {
			if (j < 0 || j > w - 1) continue;
			int id_ = id;
			if (occlusions[i])
				id_ += 128;
			if (pts[i].Y > bottom_y)
				id_ += 64;
			pbyte[j * 3 + 2] = id_;

			if (128 <= pbyte[j * 3 + 1])
				pbyte[j * 3 + 1] = green + 128;
			else
				pbyte[j * 3 + 1] = green;

			if (128 <= pbyte[j * 3])
				pbyte[j * 3] = blue + 128;
			else if (64 <= pbyte[j * 3])
				pbyte[j * 3] = blue + 64;
			else
				pbyte[j * 3] = blue;
		}
		if (i) {
			double r_prev = line.spline_ry_model_(pts[i - 1].Y);
			double overlap_ratio = (std::min(prev_lx, rx) - std::max(prev_lx, lx)) /
				(double)(std::max(prev_rx, rx) - std::min(prev_lx, lx));
			if (overlap_ratio <= 0) {
				if (prev_lx > rx) {
					for (int j = prev_lx; j <= rx; j++) {
						if (j < 0 || j > w - 1) continue;
						int id_ = id;
						if (occlusions[i])
							id_ += 128;
						if (pts[i].Y > bottom_y)
							id_ += 64;
						pbyte[j * 3 + 2] = id_;

						if (128 <= pbyte[j * 3 + 1])
							pbyte[j * 3 + 1] = green + 128;
						else
							pbyte[j * 3 + 1] = green;

						if (128 <= pbyte[j * 3])
							pbyte[j * 3] = blue + 128;
						else if (64 <= pbyte[j * 3])
							pbyte[j * 3] = blue + 64;
						else
							pbyte[j * 3] = blue;
					}
				}
				else {
					for (int j = prev_lx; j <= rx; j++) {
						if (j < 0 || j > w - 1) continue;
						int id_ = id;
						if (occlusions[i])
							id_ += 128;
						if (pts[i].Y > bottom_y)
							id_ += 64;
						pbyte[j * 3 + 2] = id_;

						if (128 <= pbyte[j * 3 + 1])
							pbyte[j * 3 + 1] = green + 128;
						else
							pbyte[j * 3 + 1] = green;

						if (128 <= pbyte[j * 3])
							pbyte[j * 3] = blue + 128;
						else if (64 <= pbyte[j * 3])
							pbyte[j * 3] = blue + 64;
						else
							pbyte[j * 3] = blue;
					}
				}
			}
		}
		prev_lx = lx;
		prev_rx = rx;
		pbyte += bmData_o.Stride;
	}
	mask.UnlockBits(&bmData_o);
}
LineTypes GetLineTypes(LaneInfo info, int id) {
	int typePos = 0;
	int typeShape = 0;
	int typeSD = 0;
	int typeColor = 0;
	int typeBicycle = 0;

	if (info.GetType4() == LaneInfo::OPPOSITE_SIDE && info.GetType3() == LaneInfo::LEFT)
		typePos = 0;
	else if (info.GetType4() == LaneInfo::BRANCH)
		typePos = 1;
	else if (info.GetType4() == LaneInfo::C4_NONE && info.GetType3() == LaneInfo::LEFT && info.GetType3_ID() == 3)
		typePos = 2;
	else if (info.GetType4() == LaneInfo::C4_NONE && info.GetType3() == LaneInfo::LEFT && info.GetType3_ID() == 2)
		typePos = 3;
	else if (info.GetType4() == LaneInfo::C4_NONE && info.GetType3() == LaneInfo::LEFT && info.GetType3_ID() == 1)
		typePos = 4;
	else if (info.GetType4() == LaneInfo::C4_NONE && info.GetType3() == LaneInfo::LEFT && info.GetType3_ID() == 0)
		typePos = 5;
	else if (info.GetType4() == LaneInfo::C4_NONE && info.GetType3() == LaneInfo::RIGHT && info.GetType3_ID() == 0)
		typePos = 6;
	else if (info.GetType4() == LaneInfo::C4_NONE && info.GetType3() == LaneInfo::RIGHT && info.GetType3_ID() == 1)
		typePos = 7;
	else if (info.GetType4() == LaneInfo::C4_NONE && info.GetType3() == LaneInfo::RIGHT && info.GetType3_ID() == 2)
		typePos = 8;
	else if (info.GetType4() == LaneInfo::C4_NONE && info.GetType3() == LaneInfo::RIGHT && info.GetType3_ID() == 3)
		typePos = 9;
	else if (info.GetType4() == LaneInfo::MERGED)
		typePos = 10;
	else if (info.GetType4() == LaneInfo::OPPOSITE_SIDE && info.GetType3() == LaneInfo::RIGHT)
		typePos = 11;
	else  //uncertain
		typePos = 12;

	typeShape = info.GetType1() - 1;
	typeSD = info.GetType2() - 1;
	typeColor = info.GetType5() - 1;
	typeBicycle = info.GetType6() - 1;

	LineTypes line_type;
	line_type.id = id;
	line_type.typeShape = typeShape;
	line_type.typeSD = typeSD;
	line_type.typePos = typePos;
	line_type.typeColor = typeColor;
	line_type.typeBicycle = typeBicycle;

	return line_type;
}

BoundaryTypes GetBoundaryTypes(BoundaryInfo info, int id, int max_level) {
	int boundary_typePos = 0;
	int boundary_typeShape = 0;

	// set boundary typePos
	if (max_level == 0) {
		if (info.GetType3() == BoundaryInfo::LEFT && info.GetType3_ID() == 0)
			boundary_typePos = 0;
		else if (info.GetType3() == BoundaryInfo::RIGHT && info.GetType3_ID() == 0)
			boundary_typePos = 1;
		else
			boundary_typePos = 2;
	}
	else if (max_level == 1) {
		if (info.GetType3() == BoundaryInfo::LEFT && info.GetType3_ID() == 1)
			boundary_typePos = 0;
		else if (info.GetType3() == BoundaryInfo::LEFT && info.GetType3_ID() == 0)
			boundary_typePos = 1;
		else if (info.GetType3() == BoundaryInfo::RIGHT && info.GetType3_ID() == 0)
			boundary_typePos = 2;
		else if (info.GetType3() == BoundaryInfo::RIGHT && info.GetType3_ID() == 1)
			boundary_typePos = 3;
		else
			boundary_typePos = 4;
	}
	else if (max_level == 2) {
		if (info.GetType3() == BoundaryInfo::LEFT && info.GetType3_ID() == 2)
			boundary_typePos = 0;
		else if (info.GetType3() == BoundaryInfo::LEFT && info.GetType3_ID() == 1)
			boundary_typePos = 1;
		else if (info.GetType3() == BoundaryInfo::LEFT && info.GetType3_ID() == 0)
			boundary_typePos = 2;
		else if (info.GetType3() == BoundaryInfo::RIGHT && info.GetType3_ID() == 0)
			boundary_typePos = 3;
		else if (info.GetType3() == BoundaryInfo::RIGHT && info.GetType3_ID() == 1)
			boundary_typePos = 4;
		else if (info.GetType3() == BoundaryInfo::RIGHT && info.GetType3_ID() == 2)
			boundary_typePos = 5;
		else
			boundary_typePos = 6;
	}
	else {
		if (info.GetType3() == BoundaryInfo::LEFT && info.GetType3_ID() > 2)
			boundary_typePos = 0;
		else if (info.GetType3() == BoundaryInfo::LEFT && info.GetType3_ID() == 2)
			boundary_typePos = 1;
		else if (info.GetType3() == BoundaryInfo::LEFT && info.GetType3_ID() == 1)
			boundary_typePos = 2;
		else if (info.GetType3() == BoundaryInfo::LEFT && info.GetType3_ID() == 0)
			boundary_typePos = 3;
		else if (info.GetType3() == BoundaryInfo::RIGHT && info.GetType3_ID() == 0)
			boundary_typePos = 4;
		else if (info.GetType3() == BoundaryInfo::RIGHT && info.GetType3_ID() == 1)
			boundary_typePos = 5;
		else if (info.GetType3() == BoundaryInfo::RIGHT && info.GetType3_ID() == 2)
			boundary_typePos = 6;
		else if (info.GetType3() == BoundaryInfo::RIGHT && info.GetType3_ID() > 2)
			boundary_typePos = 7;
		else
			boundary_typePos = 8;
	}

	//WALLS, STATIONARY_VEHICLES, GUARDRAIL, STATIC_LANE_SEPARATOR, CURBS, LANE_SEPARATOR, PLASTIC_WALL, DRUM, BEACONS, CONE, ROAD_EDGE, INNER_PARKINGSPACE, UNEXEPLAINABLE, BOUNDARY_STRUCTURE_ETCS, BOUNDARY_ETCS

	// set boundary typeShape
	if (info.GetBoundaryType() == BoundaryInfo::WALLS)
		boundary_typeShape = 0;
	else if (info.GetBoundaryType() == BoundaryInfo::STATIONARY_VEHICLES)
		boundary_typeShape = 1;
	else if (info.GetBoundaryType() == BoundaryInfo::GUARDRAIL)
		boundary_typeShape = 2;
	else if (info.GetBoundaryType() == BoundaryInfo::CURBS)
		boundary_typeShape = 3;
	else if (info.GetBoundaryType() == BoundaryInfo::STATIC_LANE_SEPARATOR ||
		info.GetBoundaryType() == BoundaryInfo::LANE_SEPARATOR)
		boundary_typeShape = 4;
	else if (info.GetBoundaryType() == BoundaryInfo::PLASTIC_WALL ||
		info.GetBoundaryType() == BoundaryInfo::DRUM ||
		info.GetBoundaryType() == BoundaryInfo::BEACONS ||
		info.GetBoundaryType() == BoundaryInfo::CONE)
		boundary_typeShape = 5;
	else if (info.GetBoundaryType() == BoundaryInfo::ROAD_EDGE)
		boundary_typeShape = 6;
	else
		boundary_typeShape = 7;

	BoundaryTypes boundary_type;
	boundary_type.id = id;
	boundary_type.typeShape = boundary_typeShape;
	boundary_type.typePos = boundary_typePos;
	return boundary_type;
}

bool WriteTypeFile(const char *szpath, vector<LineTypes> &line_types, vector<BoundaryTypes> &boundary_types, int width, int height)
{
	tinyxml2::XMLDocument xmlDoc;

	XMLNode * pRoot = xmlDoc.NewElement("LaneBoundaryTypes");
	xmlDoc.InsertFirstChild(pRoot);

	XMLElement *pElement = xmlDoc.RootElement();
	pElement->SetAttribute("imageWidth", width);     // 이미지 가로 크기
	pElement->SetAttribute("imageHeight", height);   // 이미지 세로 크기

	pElement = xmlDoc.NewElement("LaneLines");		 // LaneLines	
	for (int i = 0; i < line_types.size(); i++) {
		XMLElement * pSplineElement = xmlDoc.NewElement("LaneLine");     // LaneLine
		pSplineElement->SetAttribute("id", line_types[i].id);
		pSplineElement->SetAttribute("typeShape", line_types[i].typeShape);
		pSplineElement->SetAttribute("typeSD", line_types[i].typeSD);
		pSplineElement->SetAttribute("typePos", line_types[i].typePos);
		pSplineElement->SetAttribute("typeColor", line_types[i].typeColor);
		pSplineElement->SetAttribute("typeBicycle", line_types[i].typeBicycle);

		pElement->InsertEndChild(pSplineElement);				  	  // LaneLine 추가
	}

	pElement->SetAttribute("LaneLineNum", (int)line_types.size());
	pRoot->InsertEndChild(pElement);				  // LaneLines 추가

	pElement = xmlDoc.NewElement("BoundaryLines");		 // BoundaryLines	
	for (int i = 0; i < boundary_types.size(); i++) {
		XMLElement * pSplineElement = xmlDoc.NewElement("BoundaryLine");     // BoundaryLine
		pSplineElement->SetAttribute("id", boundary_types[i].id);
		pSplineElement->SetAttribute("typeShape", boundary_types[i].typeShape);
		pSplineElement->SetAttribute("typePos", boundary_types[i].typePos);

		pElement->InsertEndChild(pSplineElement);				  	  // BoundaryLine 추가
	}

	pElement->SetAttribute("BoundaryLineNum", (int)boundary_types.size());
	pRoot->InsertEndChild(pElement);				  // BoundaryLines 추가

	XMLError eResult = xmlDoc.SaveFile(szpath); // 파일 저장
	if (eResult != XMLError::XML_SUCCESS)
	{
		//printf("파일 저장 실패.\n");
		return 0;
	}
	return true;
}
int CPointingToolView::DrawLaneBoundaryToMask(RoadLaneManager &road, Bitmap &mask, char *xml_outpath, bool use_acc) {
	/*
	// R : Lane
	// G : Boundary
	// B : VP
	*/

	// lane
	vector<LaneLine> lines;
	vector<LaneLine> acc_lines;
	vector<LineTypes> line_types;
	for (int i = 0; i < road.GetSizeLaneLine(); i++) {
		LaneInfo info = road.line_ptr(i)->info;
		if (LaneInfo::SOLID <= info.GetType1() && info.GetType1() <= LaneInfo::CATS_EYE
			&& LaneInfo::SINGLE <= info.GetType2() && info.GetType2() <= LaneInfo::ACCESSORIE
			&& LaneInfo::LEFT <= info.GetType3() && info.GetType3() <= LaneInfo::UNCERTAIN
			&& LaneInfo::C4_NONE <= info.GetType4() && info.GetType4() <= LaneInfo::OPPOSITE_SIDE
			&& LaneInfo::WHITE <= info.GetType5() && info.GetType5() <= LaneInfo::ETC)
		{
			if (info.GetType4() != LaneInfo::UNABLE)
			{
				if (!use_acc && info.GetType2() == LaneInfo::ACCESSORIE)
					acc_lines.push_back(*(road.line_ptr(i)));
				else
					lines.push_back(*(road.line_ptr(i)));
			}
		}
	}

	// boundary
	float boundary_width_ratio = 128.0;
	float boundary_width = road.GetImageW() / (float)boundary_width_ratio;
	int boundary_max_level = 1;

	vector<BoundaryLine> boundaries;
	vector<BoundaryTypes> boundary_types;

	// get left_id, right_id
	int left_id = INT_MAX, right_id = INT_MAX;
	for (int i = 0; i < road.GetSizeBoundary(); i++) {
		BoundaryInfo boundary_info = road.boundary_ptr(i)->info;
		if ((BoundaryInfo::WALLS <= boundary_info.GetBoundaryType() && boundary_info.GetBoundaryType() <= BoundaryInfo::ROAD_EDGE) ||
			(BoundaryInfo::BOUNDARY_STRUCTURE_ETCS <= boundary_info.GetBoundaryType() && boundary_info.GetBoundaryType() <= BoundaryInfo::BOUNDARY_ETCS)) {
			int type3_id = boundary_info.GetType3_ID();
			if (type3_id >= 0 && type3_id <= boundary_max_level) {
				if (boundary_info.GetType3() == BoundaryInfo::LEFT && type3_id < left_id) {
					left_id = type3_id;
				}
				else if (boundary_info.GetType3() == BoundaryInfo::RIGHT && type3_id < right_id) {
					right_id = type3_id;
				}
			}
		}
	}
	// add to boundaries vector
	for (int i = 0; i < road.GetSizeBoundary(); i++) {
		BoundaryInfo boundary_info = road.boundary_ptr(i)->info;
		if ((BoundaryInfo::WALLS <= boundary_info.GetBoundaryType() && boundary_info.GetBoundaryType() <= BoundaryInfo::ROAD_EDGE) ||
			(BoundaryInfo::BOUNDARY_STRUCTURE_ETCS <= boundary_info.GetBoundaryType() && boundary_info.GetBoundaryType() <= BoundaryInfo::BOUNDARY_ETCS)) {
			if ((boundary_info.GetType3() == BoundaryInfo::LEFT && boundary_info.GetType3_ID() == left_id) || (boundary_info.GetType3() == BoundaryInfo::RIGHT && boundary_info.GetType3_ID() == right_id))
				boundaries.push_back(*(road.boundary_ptr(i)));
		}
		else {
			printf("this image has boundary type problem\n");
			//return -1;
		}
	}


	if (!use_acc) {
		// acc lines as road marker
		for (int i = 0; i < acc_lines.size(); i++) {
			DrawLineAsRoadMarkerToMask(acc_lines[i], mask);
		}
	}

	// draw line seg mask in R channel
	for (int i = 0; i < lines.size(); i++) {
		int line_id = i + 1;
		DrawLineSegToMask(lines[i], line_id, mask);
		line_types.push_back(GetLineTypes(lines[i].info, line_id));
	}

	// draw boundary seg mask in G channel
	int left_count = 0;
	int right_count = 0;
	for (int i = 0; i < boundaries.size(); i++) {
		int boundary_id = -1;
		if (boundaries[i].info.GetType3() == BoundaryInfo::LEFT) {
			boundary_id = (2 * left_count++) + 1; // odd
		}
		else {
			boundary_id = 2 * (1 + right_count++); // even
		}
		DrawBoundarySegToMask(boundaries[i], boundary_id, mask, boundary_width);
		boundary_types.push_back(GetBoundaryTypes(boundaries[i].info, boundary_id, boundary_max_level));
	}

	// save line, boundary type info in xml file
	int w = mask.GetWidth(), h = mask.GetHeight();
	WriteTypeFile(xml_outpath, line_types, boundary_types, w, h);

	return 0;
}

int CPointingToolView::DrawLineSegToMask(LaneLine &line, int id, Bitmap &mask) {
	int w = mask.GetWidth(), h = mask.GetHeight();
	Rect rect(0, 0, w, h);

	Gdiplus::BitmapData bmData_o;
	mask.LockBits(&rect, Gdiplus::ImageLockModeWrite, PixelFormat24bppRGB, &bmData_o);
	vector<PointF> pts;
	std::vector<PPOINTF> xy;
	std::vector<PPOINTF> wy;
	std::vector<bool> occlusions;

	int top_y = std::max((int)std::round(line.top_y_), 0);
	int bottom_y = std::min((int)std::round(line.bottom_y_), h - 1);

	bool b_ext = false;
	if (line.info.GetType4() == LaneInfo::C4_NONE && line.info.GetType3() == LaneInfo::LEFT && (line.info.GetType3_ID() == 1 || line.info.GetType3_ID() == 0)
		|| line.info.GetType4() == LaneInfo::C4_NONE && line.info.GetType3() == LaneInfo::RIGHT && (line.info.GetType3_ID() == 0 || line.info.GetType3_ID() == 1))
		b_ext = true;
	//bool b_ext = (4 <= typePos && typePos <= 7);
	for (int y = top_y; y <= (b_ext ? (h - 1) : bottom_y); y++) {
		bool occlusion = false;
		for (int k = 0; k < line.info.occlusions_top_bottom_.size(); k++) {
			if (y >= line.info.occlusions_top_bottom_[k].first &&
				y <= line.info.occlusions_top_bottom_[k].second) {
				occlusion = true;
				break;
			}
		}
		occlusions.push_back(occlusion);
		double ix = line.spline_xy_model_(y);
		pts.push_back(PPOINTF(ix, y));
	}
	int y = top_y;
	BYTE *pbyte = (BYTE *)bmData_o.Scan0 + y * bmData_o.Stride;
	int prev_lx, prev_rx;
	for (int i = 0; i < pts.size(); i++) {
		double r = line.spline_ry_model_(pts[i].Y);
		int lx = std::round(pts[i].X - r);
		int rx = std::round(pts[i].X + r);
		for (int j = lx; j <= rx; j++) {
			if (j < 0 || j > w - 1) continue;
			int id_ = id;
			if (occlusions[i])
				id_ += 128;
			if (pts[i].Y > bottom_y)
				id_ += 64;
			pbyte[j * 3 + 2] = id_;
		}
		if (i) {
			double r_prev = line.spline_ry_model_(pts[i - 1].Y);
			double overlap_ratio = (std::min(prev_lx, rx) - std::max(prev_lx, lx)) /
				(double)(std::max(prev_rx, rx) - std::min(prev_lx, lx));
			if (overlap_ratio <= 0) {
				if (prev_lx > rx) {
					for (int j = prev_lx; j <= rx; j++) {
						if (j < 0 || j > w - 1) continue;
						int id_ = id;
						if (occlusions[i])
							id_ += 128;
						if (pts[i].Y > bottom_y)
							id_ += 64;
						pbyte[j * 3 + 2] = id_;
					}
				}
				else {
					for (int j = prev_lx; j <= rx; j++) {
						if (j < 0 || j > w - 1) continue;
						int id_ = id;
						if (occlusions[i])
							id_ += 128;
						if (pts[i].Y > bottom_y)
							id_ += 64;
						pbyte[j * 3 + 2] = id_;
					}
				}
			}
		}
		prev_lx = lx;
		prev_rx = rx;
		pbyte += bmData_o.Stride;
	}
	mask.UnlockBits(&bmData_o);
	return 0;
}
