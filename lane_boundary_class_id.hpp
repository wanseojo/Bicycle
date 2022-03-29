#ifndef _LANE_BOUNDARY_CLASS_ID_HPP_
#define _LANE_BOUNDARY_CLASS_ID_HPP_

#include "sv_types.h"

namespace sv {

static const int32_t LDA_MAX_TYPE_NUM = 10;
static const int32_t LDA_MAX_CLASS_NUM = 20;

enum LaneBoundaryClassID {
    LDA_CLASS_ID_MIN = 0,

    LDA_CLASS_ID_Undefine,

    LDA_CLASS_ID_PATTERN_Unknown,
    LDA_CLASS_ID_PATTERN_Solid,
    LDA_CLASS_ID_PATTERN_Dashed,
    LDA_CLASS_ID_PATTERN_BottsDot,
    LDA_CLASS_ID_PATTERN_Fishbone,
    LDA_CLASS_ID_PATTERN_Zigzag,

    LDA_CLASS_ID_COLOR_Unknown,
    LDA_CLASS_ID_COLOR_White,
    LDA_CLASS_ID_COLOR_Yellow,
    LDA_CLASS_ID_COLOR_Blue,
    LDA_CLASS_ID_COLOR_Red,
    LDA_CLASS_ID_COLOR_Green,

    LDA_CLASS_ID_ROADBOUNDARY_Unknown,
    LDA_CLASS_ID_ROADBOUNDARY_Edge,
    LDA_CLASS_ID_ROADBOUNDARY_Guardrail,
    LDA_CLASS_ID_ROADBOUNDARY_Barriers,
    LDA_CLASS_ID_ROADBOUNDARY_Curb,
    LDA_CLASS_ID_ROADBOUNDARY_Wall,
    LDA_CLASS_ID_ROADBOUNDARY_ParkedCars,
    LDA_CLASS_ID_ROADBOUNDARY_Beacons,
    LDA_CLASS_ID_ROADBOUNDARY_TrafficCones,
    LDA_CLASS_ID_ROADBOUNDARY_LaneSeparator,

    LDA_CLASS_ID_POSITION_Unknown,
    LDA_CLASS_ID_POSITION_EgoLeft,
    LDA_CLASS_ID_POSITION_NeighborLeft,
    LDA_CLASS_ID_POSITION_NeighborLeftLeft,
    LDA_CLASS_ID_POSITION_NeighborLeftLeftLeft,
    LDA_CLASS_ID_POSITION_LeftOpposite,
    LDA_CLASS_ID_POSITION_EgoRight,
    LDA_CLASS_ID_POSITION_NeighborRight,
    LDA_CLASS_ID_POSITION_NeighborRightRight,
    LDA_CLASS_ID_POSITION_NeighborRightRightRight,
    LDA_CLASS_ID_POSITION_RightOpposite,

    LDA_CLASS_ID_MULTIPLE_LINE_Unknown,
    LDA_CLASS_ID_MULTIPLE_LINE_Single,
    LDA_CLASS_ID_MULTIPLE_LINE_Double,
    LDA_CLASS_ID_MULTIPLE_LINE_Multiple,
    LDA_CLASS_ID_MULTIPLE_LINE_Accessory,
    LDA_CLASS_ID_MULTIPLE_LINE_Undefine,

    LDA_CLASS_ID_BRANCH_Unknown,
    LDA_CLASS_ID_BRANCH,
    LDA_CLASS_ID_BRANCH_Exit,

    LDA_CLASS_ID_MERGED_Unknown,
    LDA_CLASS_ID_MERGED,
    LDA_CLASS_ID_MERGED_Entrance,

    LDA_CLASS_ID_TRAFFIC_DIRECTION_Unknown,
    LDA_CLASS_ID_TRAFFIC_DIRECTION_Oncomming,
    LDA_CLASS_ID_TRAFFIC_DIRECTION_Preceding,

    LDA_CLASS_ID_LINE_VALIDITY_Unknown,
    LDA_CLASS_ID_LINE_VALIDITY_Valid,
    LDA_CLASS_ID_LINE_VALIDITY_Invalid,

	LDA_CLASS_ID_BICYCLE_IS,
	LDA_CLASS_ID_BICYCLE_NOT,

    LDA_CLASS_ID_MAX = 64
};
}

#endif