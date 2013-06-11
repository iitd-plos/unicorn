
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

struct Color
{
    std::string htmlRep; // #00FFCC

    Color();
};

struct Line
{
    Color color;
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

class Graph
{
public:
    Graph(size_t pWidth, size_t pHeight, std::auto_ptr<Axis>& pAxisX, std::auto_ptr<Axis>& pAxisY);

    size_t GetWidth() const;
    size_t GetHeight() const;
    
    virtual const std::string& GetSvg() = 0;

protected:
    void GetPreSvg();
    void GetPostSvg(size_t pMaxDataPoints);

    double mWidth, mHeight;
    double mLeftMargin, mRightMargin, mTopMargin, mBottomMargin;
    double mUsableWidth, mUsableHeight;

    std::auto_ptr<Axis> mAxisX;
    std::auto_ptr<Axis> mAxisY;

    double mMinX, mMinY, mMaxX, mMaxY;
    double mMinPlottedX, mMinPlottedY;
    double mPlottedSpaceX, mPlottedSpaceY;
    double mPixelsPerUnitX, mPixelsPerUnitY;
    
    std::string mSvg;
};

class LineGraph : public Graph
{
public:
    LineGraph(size_t pWidth, size_t pHeight, std::auto_ptr<Axis>& pAxisX, std::auto_ptr<Axis>& pAxisY, size_t pLineCount);
    
    void SetLineName(size_t pLineIndex, const std::string& pName);
    void AddLineDataPoint(size_t pLineIndex, const std::pair<double, double>& pDataPoint);
    
    virtual const std::string& GetSvg();
    
private:
    std::vector<Line> mLines;
};

class RectGraph : public Graph
{
public:
    RectGraph(size_t pWidth, size_t pHeight, std::auto_ptr<Axis>& pAxisX, std::auto_ptr<Axis>& pAxisY, size_t pGroups, size_t pRectsPerGroup, bool pGroupsOnXAxis = true);

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

#endif





