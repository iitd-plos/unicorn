    
/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institue of Technology, New Delhi. Redistribution,
 * modification and any use in source form is strictly prohibited
 * without formal written approval from Indian Institute of Technology,
 * New Delhi. Use of software in binary form is allowed provided
 * the using application clearly highlights the credits.
 *
 * This work is the doctoral project of Tarun Beri under the guidance
 * of Prof. Subodh Kumar and Prof. Sorav Bansal. More information
 * about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 */

#include <iostream>
#include <iomanip>

#include <sstream>
#include <limits>
#include <cstdlib>
#include <time.h>

#include "graph.h"

#define LEFT_MARGIN_PERCENTAGE 18
#define RIGHT_MARGIN_PERCENTAGE 5
#define TOP_MARGIN_PERCENTAGE 5
#define BOTTOM_MARGIN_PERCENTAGE 20
#define MAX_MAJOR_TICKS_X 10
#define MAX_MAJOR_TICKS_Y 10
#define LEGEND_MARGIN_PERCENTAGE 2
#define INTER_LEGEND_HORIZONTAL_SPACING_PERCENTAGE 1
#define INTER_LEGEND_VERTICAL_SPACING_PERCENTAGE 3
#define MAX_LEGEND_ROWS 2
#define MAX_LEGEND_COLS 4
#define RECT_GROUPS_SPACING_PERCENTAGE 5
#define AXIS_STROKE_WIDTH 4     // Should be divisible by 2
#define TICK_LENGTH 4;

const char* const gColors[] =
{
    "#000000", "#000033", "#000066", "#000099", "#0000CC", "#0000FF",
    "#003300", "#003333", "#003366", "#003399", "#0033CC", "#0033FF",
    "#006600", "#006633", "#006666", "#006699", "#0066CC", "#0066FF",
    "#009900", "#009933", "#009966", "#009999", "#0099CC", "#0099FF",
    "#00CC00", "#00CC33", "#00CC66", "#00CC99", "#00CCCC", "#00CCFF",
    "#00FF00", "#00FF33", "#00FF66", "#00FF99", "#00FFCC", "#00FFFF",
    "#330000", "#330033", "#330066", "#330099", "#3300CC", "#3300FF",
    "#333300", "#333333", "#333366", "#333399", "#3333CC", "#3333FF",
    "#336600", "#336633", "#336666", "#336699", "#3366CC", "#3366FF",
    "#339900", "#339933", "#339966", "#339999", "#3399CC", "#3399FF",
    "#33CC00", "#33CC33", "#33CC66", "#33CC99", "#33CCCC", "#33CCFF",
    "#33FF00", "#33FF33", "#33FF66", "#33FF99", "#33FFCC", "#33FFFF",
    "#660000", "#660033", "#660066", "#660099", "#6600CC", "#6600FF",
    "#663300", "#663333", "#663366", "#663399", "#6633CC", "#6633FF",
    "#666600", "#666633", "#666666", "#666699", "#6666CC", "#6666FF",
    "#669900", "#669933", "#669966", "#669999", "#6699CC", "#6699FF",
    "#66CC00", "#66CC33", "#66CC66", "#66CC99", "#66CCCC", "#66CCFF",
    "#66FF00", "#66FF33", "#66FF66", "#66FF99", "#66FFCC", "#66FFFF",
    "#990000", "#990033", "#990066", "#990099", "#9900CC", "#9900FF",
    "#993300", "#993333", "#993366", "#993399", "#9933CC", "#9933FF",
    "#996600", "#996633", "#996666", "#996699", "#9966CC", "#9966FF",
    "#999900", "#999933", "#999966", "#999999", "#9999CC", "#9999FF",
    "#99CC00", "#99CC33", "#99CC66", "#99CC99", "#99CCCC", "#99CCFF",
    "#99FF00", "#99FF33", "#99FF66", "#99FF99", "#99FFCC", "#99FFFF",
    "#CC0000", "#CC0033", "#CC0066", "#CC0099", "#CC00CC", "#CC00FF",
    "#CC3300", "#CC3333", "#CC3366", "#CC3399", "#CC33CC", "#CC33FF",
    "#CC6600", "#CC6633", "#CC6666", "#CC6699", "#CC66CC", "#CC66FF",
    "#CC9900", "#CC9933", "#CC9966", "#CC9999", "#CC99CC", "#CC99FF",
    "#CCCC00", "#CCCC33", "#CCCC66", "#CCCC99", "#CCCCCC", "#CCCCFF",
    "#CCFF00", "#CCFF33", "#CCFF66", "#CCFF99", "#CCFFCC", "#CCFFFF",
    "#FF0000", "#FF0033", "#FF0066", "#FF0099", "#FF00CC", "#FF00FF",
    "#FF3300", "#FF3333", "#FF3366", "#FF3399", "#FF33CC", "#FF33FF",
    "#FF6600", "#FF6633", "#FF6666", "#FF6699", "#FF66CC", "#FF66FF",
    "#FF9900", "#FF9933", "#FF9966", "#FF9999", "#FF99CC", "#FF99FF",
    "#FFCC00", "#FFCC33", "#FFCC66", "#FFCC99", "#FFCCCC", "#FFCCFF"
};


/* struct Color */
Color::Color()
{
}


/* struct Axis */
Axis::Axis(const std::string& pLabel, bool pShowMajorTicks /* = true */)
: label(pLabel)
, showMajorTicks(pShowMajorTicks)
{
}


/* struct Rect */
Rect::Rect(double pMinX, double pMinY, double pMaxX, double pMaxY)
: minX(pMinX)
, minY(pMinY)
, maxX(pMaxX)
, maxY(pMaxY)
{
}


/* struct RectGroup */
RectGroup::RectGroup()
: minX(std::numeric_limits<double>::infinity())
, minY(std::numeric_limits<double>::infinity())
, maxX(0)
, maxY(0)
{
}



/* class Graph */
Graph::Graph(size_t pWidth, size_t pHeight, std::auto_ptr<Axis>& pAxisX, std::auto_ptr<Axis>& pAxisY)
: mWidth(pWidth)
, mHeight(pHeight)
, mLeftMargin(mWidth * LEFT_MARGIN_PERCENTAGE/100.0)
, mRightMargin(mWidth * RIGHT_MARGIN_PERCENTAGE/100.0)
, mTopMargin(mHeight * TOP_MARGIN_PERCENTAGE/100.0)
, mBottomMargin(mHeight * BOTTOM_MARGIN_PERCENTAGE/100.0)
, mUsableWidth(mWidth - mLeftMargin - mRightMargin)
, mUsableHeight(mHeight - mTopMargin - mBottomMargin)
, mAxisX(pAxisX)
, mAxisY(pAxisY)
, mMinX(std::numeric_limits<double>::infinity())
, mMinY(std::numeric_limits<double>::infinity())
, mMaxX(0.0)
, mMaxY(0.0)
, mMinPlottedX(0.0)
, mMinPlottedY(0.0)
, mPlottedSpaceX(0.0)
, mPlottedSpaceY(0.0)
, mPixelsPerUnitX(0.0)
, mPixelsPerUnitY(0.0)
{
}

void Graph::GetPreSvg()
{
    mMinPlottedX = mMinX;
    mMinPlottedY = 0;    //mMinY;
    
    mPlottedSpaceX = (mMaxX - mMinPlottedX);
    mPlottedSpaceY = (mMaxY - mMinPlottedY);
    
    mPixelsPerUnitX = mUsableWidth / mPlottedSpaceX;
    mPixelsPerUnitY = mUsableHeight / mPlottedSpaceY;
}

void Graph::GetPostSvg(size_t pMaxDataPoints)
{
    std::stringstream lStream;
    
    size_t lTickLength = TICK_LENGTH;
    size_t lTickLabelsPixelAdjustment = 3;
    
    if(mAxisX.get())
    {
        lStream << "<line x1='" << mLeftMargin << "' y1='" << mHeight - (mBottomMargin - AXIS_STROKE_WIDTH/2) << "'";
        lStream << " x2='" << mLeftMargin + mUsableWidth << "' y2='" << mHeight - (mBottomMargin - AXIS_STROKE_WIDTH/2) << "'";
        lStream << " style='stroke:black; stroke-width:" << AXIS_STROKE_WIDTH << "'/>" << std::endl;
        
        lStream << std::fixed << std::setprecision(2);

        if(mAxisX->showMajorTicks)
        {
            size_t lMajorIntervals = std::min((size_t)MAX_MAJOR_TICKS_X - 1, pMaxDataPoints);
            size_t lMajorTicksX = lMajorIntervals + 1;
            double lMajorTicksSpacing = (mPlottedSpaceX / lMajorIntervals);
            double lMajorTicksPixels = lMajorTicksSpacing * mPixelsPerUnitX;

            for(size_t i = 0; i < lMajorTicksX; ++i)
            {
                double lPos = ((i != lMajorTicksX - 1) ? (mLeftMargin + i * lMajorTicksPixels) : (mLeftMargin + mUsableWidth));
                double lVal = ((i != lMajorTicksX - 1) ? (mMinPlottedX + i * lMajorTicksSpacing) : mMaxX);

                lStream << "<line x1='" << lPos << "' y1='" << mHeight - (mBottomMargin - AXIS_STROKE_WIDTH) << "'";
                lStream << " x2='" << lPos << "' y2='" << mHeight -  (mBottomMargin - lTickLength - AXIS_STROKE_WIDTH) << "'";
                lStream << " style='stroke:black; stroke-width:1' />" << std::endl;
                
                double lX = lPos - lTickLabelsPixelAdjustment;
                double lY = mHeight -  (mBottomMargin - lTickLength - AXIS_STROKE_WIDTH - lTickLabelsPixelAdjustment);
                
                if((lVal - (long)lVal) == (double)0.0)
                    lStream << "<text font-size='60%' x='" << lX << "' y='" << lY << "' transform='rotate(90 " << lX << "," << lY << ")'>" << (long)lVal << "</text>" << std::endl;
                else
                    lStream << "<text font-size='60%' x='" << lX << "' y='" << lY << "' transform='rotate(90 " << lX << "," << lY << ")'>" << lVal << "</text>" << std::endl;
            }
        }
        
        if(!mAxisX->label.empty())
        {
            double lHorizPos = mLeftMargin + mUsableWidth/2.0;
            double lVerticalPos = 100.0 - LEGEND_MARGIN_PERCENTAGE - INTER_LEGEND_VERTICAL_SPACING_PERCENTAGE * (MAX_LEGEND_ROWS + 0.5);

            lStream << "<text text-anchor='middle' font-size='80%' x='" << lHorizPos << "' y='" << lVerticalPos << "%'>" << mAxisX->label << "</text>" << std::endl;
        }
    }

    if(mAxisY.get())
    {
        lStream << "<line x1='" << mLeftMargin - AXIS_STROKE_WIDTH/2 << "' y1='" << mHeight - (mBottomMargin - (mAxisX.get() ? AXIS_STROKE_WIDTH : 0)) << "'";
        lStream << " x2='" << mLeftMargin - AXIS_STROKE_WIDTH/2 << "' y2='" << mHeight -  (mBottomMargin + mUsableHeight) << "'";
        lStream << " style='stroke:black; stroke-width:" << AXIS_STROKE_WIDTH << "'/>" << std::endl;

        if(mAxisY->showMajorTicks)
        {
            size_t lMajorIntervals = std::min((size_t)MAX_MAJOR_TICKS_Y - 1, pMaxDataPoints);
            size_t lMajorTicksY = lMajorIntervals + 1;
            double lMajorTicksSpacing = (mPlottedSpaceY / lMajorIntervals);
            double lMajorTicksPixels = lMajorTicksSpacing * mPixelsPerUnitY;

            lStream << std::fixed << std::setprecision(2);
            
            for(size_t i = 0; i < lMajorTicksY; ++i)
            {
                double lPos = ((i != lMajorTicksY - 1) ? (mHeight - (mBottomMargin + i * lMajorTicksPixels)) : (mHeight - (mBottomMargin + mUsableHeight)));
                double lVal = ((i != lMajorTicksY - 1) ? (mMinPlottedY + i * lMajorTicksSpacing) : mMaxY);

                lStream << "<line x1='" << mLeftMargin - AXIS_STROKE_WIDTH << "' y1='" << lPos << "'";
                lStream << " x2='" << mLeftMargin - AXIS_STROKE_WIDTH - lTickLength << "' y2='" << lPos << "'";
                lStream << " style='stroke:black; stroke-width:1' />" << std::endl;
                
                double lX = mLeftMargin - AXIS_STROKE_WIDTH - lTickLength - lTickLabelsPixelAdjustment;
                double lY = lPos + lTickLabelsPixelAdjustment;

                if((lVal - (long)lVal) == (double)0.0)
                    lStream << "<text text-anchor='end' font-size='60%' x='" << lX << "' y='" << lY << "'>" << (long)lVal << "</text>" << std::endl;
                else
                    lStream << "<text text-anchor='end' font-size='60%' x='" << lX << "' y='" << lY << "'>" << lVal << "</text>" << std::endl;
            }
        }

        if(!mAxisY->label.empty())
        {
            double lHorizPos = mWidth * 2 * LEGEND_MARGIN_PERCENTAGE / 100.0;
            double lVerticalPos = mHeight - (mBottomMargin + mUsableHeight/2.0);

            lStream << "<text text-anchor='middle' font-size='80%' x='" << lHorizPos << "' y='" << lVerticalPos << "' transform='rotate(-90 " << lHorizPos << "," << lVerticalPos << ")' >" << mAxisY->label << "</text>" << std::endl;
        }
    }

    mSvg.append(lStream.str());
}

size_t Graph::GetWidth() const
{
    return (size_t)mWidth;
}

size_t Graph::GetHeight() const
{
    return (size_t)mHeight;
}


/* class LineGraph */
LineGraph::LineGraph(size_t pWidth, size_t pHeight, std::auto_ptr<Axis>& pAxisX, std::auto_ptr<Axis>& pAxisY, size_t pLineCount)
: Graph(pWidth, pHeight, pAxisX, pAxisY)
{
    mLines.resize(pLineCount);
    for(size_t i = 0; i < pLineCount; ++i)
        mLines[i].color.htmlRep = std::string(gColors[std::rand() % (sizeof(gColors) / sizeof(gColors[0]))]);
}

void LineGraph::SetLineName(size_t pLineIndex, const std::string &pName)
{
    if(pLineIndex >= mLines.size())
        throw std::exception();
    
    mLines[pLineIndex].name = pName;
}

void LineGraph::AddLineDataPoint(size_t pLineIndex, const std::pair<double, double>& pDataPoint)
{
    if(pLineIndex >= mLines.size())
        throw std::exception();

    mMinX = std::min(mMinX, pDataPoint.first);
    mMaxX = std::max(mMaxX, pDataPoint.first);
    mMinY = std::min(mMinY, pDataPoint.second);
    mMaxY = std::max(mMaxY, pDataPoint.second);
    
    mLines[pLineIndex].dataPoints.push_back(pDataPoint);
}

const std::string& LineGraph::GetSvg()
{
    if(!mSvg.empty())
        return mSvg;

    Graph::GetPreSvg();
    
    std::stringstream lStream;

    size_t lMaxDataPoints = 0;

    std::vector<Line>::const_iterator lIter = mLines.begin(), lEndIter = mLines.end();
    for(size_t lIndex = 0; lIter != lEndIter; ++lIter, ++lIndex)
    {
        size_t lDataPoints = 0;

        lStream << "<polyline points='";
        
        std::vector<std::pair<double, double> >::const_iterator lInnerIter = (*lIter).dataPoints.begin(), lInnerEndIter = (*lIter).dataPoints.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            double lValX = (*lInnerIter).first - mMinPlottedX;
            double lValY = (*lInnerIter).second - mMinPlottedY;
            
            lStream << mLeftMargin + lValX * mPixelsPerUnitX << "," << mHeight - (mBottomMargin + lValY * mPixelsPerUnitY) << " ";
            ++lDataPoints;
        }
        
        lStream << "' style='fill:none; stroke:" << (*lIter).color.htmlRep << "; stroke-width:1' />" << std::endl;
        lMaxDataPoints = std::max(lMaxDataPoints, lDataPoints);

        if(lIndex < MAX_LEGEND_ROWS * MAX_LEGEND_COLS)
        {
            size_t lLegendCountPerRow = std::min((size_t)MAX_LEGEND_COLS, mLines.size());
            double lLegendAdjustmentPercentage = 0.75;
            double lLegendSpace = (100.0 - 2.0 * LEGEND_MARGIN_PERCENTAGE - (lLegendCountPerRow - 1) * INTER_LEGEND_HORIZONTAL_SPACING_PERCENTAGE);
            double lSpacePerLegend = lLegendSpace / lLegendCountPerRow;
            size_t lLegendRow = lIndex / MAX_LEGEND_COLS;
            size_t lLegendCol = lIndex % MAX_LEGEND_COLS;
            
            double lX1 = LEGEND_MARGIN_PERCENTAGE + lLegendCol * (lSpacePerLegend + INTER_LEGEND_HORIZONTAL_SPACING_PERCENTAGE);
            double lY = 100.0 - LEGEND_MARGIN_PERCENTAGE - INTER_LEGEND_VERTICAL_SPACING_PERCENTAGE * lLegendRow;
            double lX2 = lX1 + lSpacePerLegend/4.0; // one-fourth space for line and three-fourth for label

            lStream << "<line x1='" << lX1 << "%' y1='" << lY << "%' x2='" << lX2 << "%' y2='" << lY << "%' style='stroke:" << (*lIter).color.htmlRep << "; stroke-width:1' />" << std::endl;
            lStream << "<text font-size='60%' x='" << lX2 + lLegendAdjustmentPercentage << "%' y='" << lY + lLegendAdjustmentPercentage << "%'>" << (*lIter).name << "</text>" << std::endl;
        }
    }

    mSvg.append(lStream.str());
    
    Graph::GetPostSvg(lMaxDataPoints);
    
    return mSvg;
}


/* class RectGraph */
RectGraph::RectGraph(size_t pWidth, size_t pHeight, std::auto_ptr<Axis>& pAxisX, std::auto_ptr<Axis>& pAxisY, size_t pGroups, size_t pRectsPerGroup, bool pGroupsOnXAxis /* = true */)
: Graph(pWidth, pHeight, pAxisX, pAxisY)
, mRectsPerGroup(pRectsPerGroup)
, mGroupsOnXAxis(pGroupsOnXAxis)
{
    if((mGroupsOnXAxis && mAxisX.get() && mAxisX->showMajorTicks) || (!mGroupsOnXAxis && mAxisY.get() && mAxisY->showMajorTicks))
        throw std::exception();

    mRectGroups.resize(pGroups);
    mRectNames.resize(mRectsPerGroup);
}

void RectGraph::SetGroupName(size_t pGroupIndex, const std::string& pGroupName)
{
    if(pGroupIndex >= mRectGroups.size())
        throw std::exception();

    mRectGroups[pGroupIndex].name = pGroupName;
}

void RectGraph::SetRectName(size_t pRectIndexInEachGroup, const std::string& pRectName)
{
    if(pRectIndexInEachGroup >= mRectNames.size())
        throw std::exception();
    
    mRectNames[pRectIndexInEachGroup] = pRectName;
}

void RectGraph::AddRect(size_t pGroupIndex, const Rect& pRect)
{
    if(pGroupIndex >= mRectGroups.size())
        throw std::exception();

    RectGroup& lGroup = mRectGroups[pGroupIndex];
    lGroup.minX = std::min(lGroup.minX, pRect.minX);
    lGroup.minY = std::min(lGroup.minY, pRect.minY);
    lGroup.maxX = std::max(lGroup.maxX, pRect.maxX);
    lGroup.maxY = std::max(lGroup.maxY, pRect.maxY);
    
    if(mGroupsOnXAxis)
    {
        mMinY = std::min(mMinY, lGroup.minY);
        mMaxY = std::max(mMaxY, lGroup.maxY);
    }
    else
    {
        mMinX = std::min(mMinX, lGroup.minX);
        mMaxX = std::max(mMaxX, lGroup.maxX);
    }
    
    lGroup.rects.push_back(pRect);
}

const std::string& RectGraph::GetSvg()
{
    if(!mSvg.empty())
        return mSvg;
        
    Graph::GetPreSvg();
    
    std::stringstream lStream;

    std::vector<Color> lRectColors;
    for(size_t i = 0; i < mRectsPerGroup; ++i)
    {
        Color lColor;
        lColor.htmlRep = std::string(gColors[std::rand() % (sizeof(gColors) / sizeof(gColors[0]))]);
        lRectColors.push_back(lColor);
    }
    
    if(mGroupsOnXAxis)
    {
        size_t lGroupPixelSpacing = mWidth * RECT_GROUPS_SPACING_PERCENTAGE/100.0;
        size_t lPixelsForRectGroups = mUsableWidth - (mRectGroups.size() * lGroupPixelSpacing);
        size_t lPixelsPerGroup = lPixelsForRectGroups / mRectGroups.size();
        
        size_t lGroupX = mLeftMargin + lGroupPixelSpacing;
        size_t lGroupY = mBottomMargin;

        std::vector<RectGroup>::const_iterator lIter = mRectGroups.begin(), lEndIter = mRectGroups.end();
        for(; lIter != lEndIter; ++lIter)
        {
            const RectGroup& lGroup = (*lIter);
            double lSpanX = lGroup.maxX - lGroup.minX;

            std::vector<Rect>::const_iterator lInnerIter = lGroup.rects.begin(), lInnerEndIter = lGroup.rects.end();
            for(size_t lRectIndex = 0; lInnerIter != lInnerEndIter; ++lInnerIter, ++lRectIndex)
            {
                const Rect& lRect = *lInnerIter;
                double lX1 = lGroupX + (lPixelsPerGroup * lRect.minX / lSpanX);
                double lX2 = lGroupX + (lPixelsPerGroup * lRect.maxX / lSpanX);
                double lY1 = mHeight - (lGroupY + (lRect.minY - mMinPlottedY) * mPixelsPerUnitY);
                double lY2 = mHeight - (lGroupY + (lRect.maxY - mMinPlottedY) * mPixelsPerUnitY);
                
                lStream << "<rect x='" << lX1 << "' y='" << lY2 << "' width='" << lX2 - lX1 << "' height='" << lY1 - lY2 << "' fill='" << lRectColors[lRectIndex].htmlRep << "' />" << std::endl;
            }

            double lLength = AXIS_STROKE_WIDTH + 2.5 * TICK_LENGTH;
            double lX = lGroupX + lPixelsPerGroup/2.0;
            double lY = mHeight -  (mBottomMargin - lLength);
            lStream << "<text text-anchor='middle' font-size='60%' x='" << lX << "' y='" << lY << "' >" << lGroup.name << "</text>" << std::endl;
            
            lGroupX += lPixelsPerGroup + lGroupPixelSpacing;
        }
    }
    else
    {
        size_t lGroupPixelSpacing = mHeight * RECT_GROUPS_SPACING_PERCENTAGE/100.0;
        size_t lPixelsForRectGroups = mUsableHeight - (mRectGroups.size() * lGroupPixelSpacing);
        size_t lPixelsPerGroup = lPixelsForRectGroups / mRectGroups.size();
        
        size_t lGroupX = mLeftMargin;
        size_t lGroupY = mBottomMargin + lGroupPixelSpacing;

        std::vector<RectGroup>::const_iterator lIter = mRectGroups.begin(), lEndIter = mRectGroups.end();
        for(; lIter != lEndIter; ++lIter)
        {
            const RectGroup& lGroup = (*lIter);
            double lSpanY = lGroup.maxY - lGroup.minY;

            std::vector<Rect>::const_iterator lInnerIter = lGroup.rects.begin(), lInnerEndIter = lGroup.rects.end();
            for(size_t lRectIndex = 0; lInnerIter != lInnerEndIter; ++lInnerIter, ++lRectIndex)
            {
                const Rect& lRect = *lInnerIter;                
                double lX1 = (lGroupX + (lRect.minX - mMinPlottedX) * mPixelsPerUnitX);
                double lX2 = (lGroupX + (lRect.maxX - mMinPlottedX) * mPixelsPerUnitX);
                double lY1 = mHeight - (lGroupY + (lPixelsPerGroup * lRect.minY / lSpanY));
                double lY2 = mHeight - (lGroupY + (lPixelsPerGroup * lRect.maxY / lSpanY));

                lStream << "<rect x='" << lX1 << "' y='" << lY2 << "' width='" << lX2 - lX1 << "' height='" << lY1 - lY2 << "' fill='" << lRectColors[lRectIndex].htmlRep << "' />" << std::endl;
            }

            double lLength = AXIS_STROKE_WIDTH + TICK_LENGTH;
            double lX = mLeftMargin - lLength;
            double lY = mHeight - (lGroupY + lPixelsPerGroup/2.0);
            lStream << "<text text-anchor='middle' font-size='60%' x='" << lX << "' y='" << lY << "' transform='rotate(-90 " << lX << "," << lY << ")' >" << lGroup.name << "</text>" << std::endl;
            
            lGroupY += lPixelsPerGroup + lGroupPixelSpacing;
        }        
    }

    std::vector<Color>::iterator lColorIter = lRectColors.begin(), lColorEndIter = lRectColors.end();
    for(size_t lLegendIndex = 0; lColorIter != lColorEndIter; ++lColorIter, ++lLegendIndex)
    {
        if(lLegendIndex < MAX_LEGEND_ROWS * MAX_LEGEND_COLS)
        {
            size_t lLegendCountPerRow = std::min((size_t)MAX_LEGEND_COLS, lRectColors.size());
            double lLegendAdjustmentPercentage = 0.75;
            double lLegendSpace = (100.0 - 2.0 * LEGEND_MARGIN_PERCENTAGE - (lLegendCountPerRow - 1) * INTER_LEGEND_HORIZONTAL_SPACING_PERCENTAGE);
            double lSpacePerLegend = lLegendSpace / lLegendCountPerRow;
            size_t lLegendRow = lLegendIndex / MAX_LEGEND_COLS;
            size_t lLegendCol = lLegendIndex % MAX_LEGEND_COLS;
            
            double lX1 = mWidth/100.0 * (LEGEND_MARGIN_PERCENTAGE + lLegendCol * (lSpacePerLegend + INTER_LEGEND_HORIZONTAL_SPACING_PERCENTAGE));
            double lY = mHeight - (mHeight/100.0 * (LEGEND_MARGIN_PERCENTAGE + INTER_LEGEND_VERTICAL_SPACING_PERCENTAGE * lLegendRow));
            double lWidth = mWidth/100.0 * (lSpacePerLegend/4.0); // one-fourth space for line and three-fourth for label
            
            double lHorizLegendAdjustment = mWidth/100.0 * lLegendAdjustmentPercentage;
            double lVerticalLegendAdjustment = mHeight/100.0 * lLegendAdjustmentPercentage;

            lStream << "<rect x='" << lX1 << "' y='" << lY - lVerticalLegendAdjustment/2.0 << "' width='" << lWidth << "' height='" << lVerticalLegendAdjustment << "' fill='" << (*lColorIter).htmlRep << "' />" << std::endl;
            lStream << "<text font-size='60%' x='" << lX1 + lWidth + lHorizLegendAdjustment << "' y='" << lY + lVerticalLegendAdjustment << "'>" << mRectNames[lLegendIndex] << "</text>" << std::endl;
        }
    }

    mSvg.append(lStream.str());
    
    Graph::GetPostSvg(mRectsPerGroup * mRectGroups.size());
    
    return mSvg;
}


