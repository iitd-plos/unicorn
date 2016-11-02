
/*
 * Copyright (c) 2016, Tarun Beri, Sorav Bansal, Subodh Kumar
 * Copyright (c) 2016 Indian Institute of Technology Delhi
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version. Any redistribution or
 * modification must retain this copyright notice and appropriately
 * highlight the credits.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * More information about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 *
 * All bug reports and enhancement requests can be sent to the following
 * email addresses -
 * onlinetarun@gmail.com
 * sbansal@cse.iitd.ac.in
 * subodh@cse.iitd.ac.in
 */

#ifndef __ANALYZER_GRAPH__
#define __ANALYZER_GRAPH__

#include <string>
#include <vector>
#include <memory>

struct Axis
{
    std::string label;
    bool showMajorTicks;
    
    Axis(const std::string& pLabel, bool pShowMajorTicks = true);
};

struct Line
{
    std::string name;
    std::vector<std::pair<double, double> > dataPoints;
};

struct Rect
{
    double minX, minY, maxX, maxY;

    Rect(double pMinX, double pMinY, double pMaxX, double pMaxY);
};

struct RectGroup
{
    std::string name;
    std::vector<Rect> rects;
    double minX, minY, maxX, maxY;
    
    RectGroup();
};

struct Gantt
{
    std::string name;
    std::vector<std::pair<size_t, std::pair<double, double>>> variantStartEnd;
    
    Gantt();
};

class Graph
{
public:
    size_t GetWidth() const;
    size_t GetHeight() const;
    
    virtual const std::string& GetSvg() = 0;

protected:
    Graph(size_t pWidth, size_t pHeight, std::unique_ptr<Axis>& pAxisX, std::unique_ptr<Axis>& pAxisY);

    void GetPreSvg();
    void GetPostSvg(size_t pMaxDataPoints);

    double mWidth, mHeight;
    double mLeftMargin, mRightMargin, mTopMargin, mBottomMargin;
    double mUsableWidth, mUsableHeight;

    std::unique_ptr<Axis> mAxisX;
    std::unique_ptr<Axis> mAxisY;

    double mMinX, mMinY, mMaxX, mMaxY;
    double mMinPlottedX, mMinPlottedY;
    double mPlottedSpaceX, mPlottedSpaceY;
    double mPixelsPerUnitX, mPixelsPerUnitY;
    
    std::string mSvg;
};

class LineGraph : public Graph
{
public:
    LineGraph(size_t pWidth, size_t pHeight, std::unique_ptr<Axis>& pAxisX, std::unique_ptr<Axis>& pAxisY, size_t pLineCount);
    
    void SetLineName(size_t pLineIndex, const std::string& pName);
    void AddLineDataPoint(size_t pLineIndex, const std::pair<double, double>& pDataPoint);
    
    virtual const std::string& GetSvg();
    
private:
    std::vector<Line> mLines;
};

class RectGraph : public Graph
{
public:
    RectGraph(size_t pWidth, size_t pHeight, std::unique_ptr<Axis>& pAxisX, std::unique_ptr<Axis>& pAxisY, size_t pGroups, size_t pRectsPerGroup, bool pGroupsOnXAxis = true);

    void SetGroupName(size_t pGroupIndex, const std::string& pGroupName);
    void SetRectName(size_t pRectIndexInEachGroup, const std::string& pRectName);
    void AddRect(size_t pGroupIndex, const Rect& pRect);
    
    virtual const std::string& GetSvg();
    
private:
    size_t mRectsPerGroup;
    bool mGroupsOnXAxis;
    std::vector<std::string> mRectNames;
    std::vector<RectGroup> mRectGroups;
};

class GanttGraph : public Graph
{
public:
    GanttGraph(size_t pWidth, size_t pHeight, std::unique_ptr<Axis>& pAxisX, std::unique_ptr<Axis>& pAxisY, size_t pGanttCount, size_t pVariantsPerGantt);

    void SetGanttName(size_t pGanttIndex, const std::string& pGanttName);
    void SetGanttVariantName(size_t pVariantIndexInEachGantt, const std::string& pVariantName);
    void AddGanttVariant(size_t pGanttIndex, size_t pVariantIndex, const std::pair<double, double>& pStartEnd);

    virtual const std::string& GetSvg();

private:
    size_t mVariantsPerGantt;
    std::vector<std::string> mGanttVariantNames;
    std::vector<Gantt> mGantts;
};

#endif





