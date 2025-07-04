//===- Event.cpp - Sharp Simulation Event Implementation ------------------===//
//
// Implementation of event types and event queue for Sharp's simulation.
//
//===----------------------------------------------------------------------===//

#include "sharp/Simulation/Event.h"
#include <algorithm>
#include <queue>

namespace sharp {
namespace sim {

// Initialize static member
EventID Event::nextID = 1;

bool Event::isReady() const {
  // Check if all dependencies are completed
  if (dependencies.size() != completedDeps.size()) {
    return false;
  }
  
  // Verify all dependency IDs match
  for (auto dep : dependencies) {
    if (std::find(completedDeps.begin(), completedDeps.end(), dep->getID()) ==
        completedDeps.end()) {
      return false;
    }
  }
  
  return true;
}

void Event::markDependencyComplete(EventID depID) {
  // Add to completed list if not already there
  if (std::find(completedDeps.begin(), completedDeps.end(), depID) ==
      completedDeps.end()) {
    completedDeps.push_back(depID);
  }
}

void EventQueue::push(EventPtr event) {
  if (event->isReady()) {
    events.push_back(event);
    std::push_heap(events.begin(), events.end(),
                   [](const EventPtr& a, const EventPtr& b) {
                     return *a < *b;
                   });
  } else {
    deferred.push_back(event);
  }
}

EventPtr EventQueue::popReady() {
  // First check deferred events
  checkDeferred();
  
  // Find next event that's ready and at or before current time
  while (!events.empty()) {
    std::pop_heap(events.begin(), events.end(),
                  [](const EventPtr& a, const EventPtr& b) {
                    return *a < *b;
                  });
    
    EventPtr next = events.back();
    events.pop_back();
    
    if (next->getTime() <= currentTime && next->isReady()) {
      return next;
    } else if (next->getTime() > currentTime) {
      // Put it back and advance time
      events.push_back(next);
      std::push_heap(events.begin(), events.end(),
                     [](const EventPtr& a, const EventPtr& b) {
                       return *a < *b;
                     });
      
      // Advance time to next event
      if (!events.empty()) {
        currentTime = (*std::min_element(events.begin(), events.end(),
                                       [](const EventPtr& a, const EventPtr& b) {
                                         return a->getTime() < b->getTime();
                                       }))->getTime();
      }
    } else {
      // Not ready, defer it
      deferred.push_back(next);
    }
  }
  
  return nullptr;
}

void EventQueue::markComplete(EventID id) {
  // Notify all deferred events about completion
  for (auto& event : deferred) {
    event->markDependencyComplete(id);
  }
  
  // Check if any deferred events are now ready
  checkDeferred();
}

void EventQueue::checkDeferred() {
  std::vector<EventPtr> stillDeferred;
  
  for (auto& event : deferred) {
    if (event->isReady() && event->getTime() <= currentTime) {
      events.push_back(event);
      std::push_heap(events.begin(), events.end(),
                     [](const EventPtr& a, const EventPtr& b) {
                       return *a < *b;
                     });
    } else {
      stillDeferred.push_back(event);
    }
  }
  
  deferred = std::move(stillDeferred);
}

} // namespace sim
} // namespace sharp