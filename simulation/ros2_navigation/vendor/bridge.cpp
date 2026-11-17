// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

// Original C ABI around the independently licensed RVO2 library.
#include "RVO2/src/RVO.h"
#include <cmath>
#include <exception>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>

namespace {
thread_local std::string error;
struct Context { RVO::RVOSimulator simulator; std::size_t count = 0; };
}

extern "C" {
int pedestrian_rvo_abi_version() { return 3; }
const char* pedestrian_rvo_error() { return error.c_str(); }
void* pedestrian_rvo_create(const double* walls, const double* velocities, int wall_count) {
    try {
        auto* context = new Context;
        for (int i=0; i<wall_count; ++i) {
            std::vector<RVO::Vector2> line{
                RVO::Vector2(walls[4*i],walls[4*i+1]),
                RVO::Vector2(walls[4*i+2],walls[4*i+3])};
            if(line[0].x()==line[1].x() && line[0].y()==line[1].y()) continue;
            const auto first=context->simulator.addObstacle(line);
            for(std::size_t j=0;j<line.size();++j)
                context->simulator.setObstacleVelocity(first+j,RVO::Vector2(velocities[2*i],velocities[2*i+1]));
        }
        context->simulator.processObstacles();
        return context;
    } catch(const std::exception& e) { error=e.what(); return nullptr; }
}
void pedestrian_rvo_destroy(void* value) { delete static_cast<Context*>(value); }
int pedestrian_rvo_step(void* value, int count, const double* positions, const double* velocities,
    const double* preferred, const double* radii, const double* speeds, const unsigned char* passive,
    double dt, double distance, double horizon, double obstacle_horizon, double pair_padding, double obstacle_padding, double* result) {
    try {
        auto& context=*static_cast<Context*>(value);
        auto& sim=context.simulator;
        if (context.count && context.count!=static_cast<std::size_t>(count)) return -2;
        sim.setTimeStep(dt);
        while(context.count<static_cast<std::size_t>(count)) {
            sim.addAgent(RVO::Vector2(),distance,count,horizon,obstacle_horizon,.25,1.,RVO::Vector2());
            ++context.count;
        }
        for(int i=0;i<count;++i) {
            sim.setAgentPosition(i,RVO::Vector2(positions[2*i],positions[2*i+1]));
            sim.setAgentVelocity(i,RVO::Vector2(velocities[2*i],velocities[2*i+1]));
            sim.setAgentPrefVelocity(i,RVO::Vector2(preferred[2*i],preferred[2*i+1]));
            sim.setAgentRadius(i,radii[i]); sim.setAgentMaxSpeed(i,speeds[i]);
            sim.setAgentPairPadding(i,pair_padding);
            sim.setAgentObstaclePadding(i,obstacle_padding);
            sim.setAgentPassive(i,passive[i]);
            sim.setAgentNeighborDist(i,distance);sim.setAgentMaxNeighbors(i,passive[i]?0:count);
            sim.setAgentTimeHorizon(i,horizon);sim.setAgentTimeHorizonObst(i,obstacle_horizon);
        }
        sim.doStep();
        for(int i=0;i<count;++i) {
            const auto velocity=sim.getAgentVelocity(i);
            result[2*i]=velocity.x();result[2*i+1]=velocity.y();
            if(!std::isfinite(result[2*i]) || !std::isfinite(result[2*i+1])) return -3;
        }
        return 0;
    } catch(const std::exception& e) {error=e.what();return -1;}
}
int pedestrian_rvo_heading(void* value, int index, double yaw, double angle, double speed,
    double preferred_x, double preferred_y, double* result) {
    auto& sim=static_cast<Context*>(value)->simulator;
    const auto count=sim.getAgentNumORCALines(index);
    std::vector<RVO::Vector2> candidates{RVO::Vector2(),RVO::Vector2(preferred_x,preferred_y)};
    auto cross=[](const RVO::Vector2& a,const RVO::Vector2& b){return a.x()*b.y()-a.y()*b.x();};
    const double desired=std::atan2(preferred_y,preferred_x);
    const double difference=std::atan2(std::sin(desired-yaw),std::cos(desired-yaw));
    for(double heading:{yaw-angle,yaw+angle,yaw,yaw+std::clamp(difference,-angle,angle)}) {
        const RVO::Vector2 direction(std::cos(heading),std::sin(heading));
        double lower=0.,upper=speed;
        for(std::size_t k=0;k<count;++k) {
            const auto& line=sim.getAgentORCALine(index,k);
            const double coefficient=cross(line.direction,direction);
            const double offset=cross(line.direction,line.point);
            if(std::abs(coefficient)<1e-10) {if(offset>1e-6) lower=std::numeric_limits<double>::infinity();}
            else if(coefficient>0) lower=std::max(lower,offset/coefficient);
            else upper=std::min(upper,offset/coefficient);
        }
        if(lower<=upper) {
            const double magnitude=std::clamp(preferred_x*direction.x()+preferred_y*direction.y(),lower,upper);
            candidates.push_back(direction*magnitude);
        }
    }
    const RVO::Vector2 preferred(preferred_x,preferred_y);
    for(std::size_t k=0;k<count;++k) {
        const auto& a=sim.getAgentORCALine(index,k);
        candidates.push_back(a.point+a.direction*((preferred-a.point)*a.direction));
        const double projection=a.point*a.direction;
        const double discriminant=projection*projection+speed*speed-RVO::absSq(a.point);
        if(discriminant>=0) for(double sign:{-1.,1.})
            candidates.push_back(a.point+a.direction*(-projection+sign*std::sqrt(discriminant)));
        for(std::size_t j=0;j<k;++j) {
            const auto& b=sim.getAgentORCALine(index,j);
            const double denominator=cross(a.direction,b.direction);
            if(std::abs(denominator)>1e-9)
                candidates.push_back(a.point+a.direction*(cross(b.point-a.point,b.direction)/denominator));
        }
    }
    double best=std::numeric_limits<double>::infinity();
    result[0]=result[1]=0.;
    for(const auto& candidate:candidates) {
        const double x=candidate.x(),y=candidate.y();
        const double magnitude_sq=x*x+y*y;
        if(speed==0. ? magnitude_sq>0. : magnitude_sq>speed*speed+1e-7) continue;
        const double heading=std::atan2(y,x);
        if(magnitude_sq>0. && std::abs(std::atan2(std::sin(heading-yaw),std::cos(heading-yaw)))>angle+1e-6) continue;
        bool valid=true;
        for(std::size_t k=0;k<count;++k) {
            const auto& line=sim.getAgentORCALine(index,k);
            if(cross(line.direction,line.point-candidate)>1e-6) {valid=false;break;}
        }
        const double distance=RVO::absSq(candidate-preferred);
        if(valid && distance<best) {best=distance;result[0]=candidate.x();result[1]=candidate.y();}
    }
    return std::isfinite(best)?0:1;
}
}
