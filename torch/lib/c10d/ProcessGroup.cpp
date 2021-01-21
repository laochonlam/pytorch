#include <c10d/ProcessGroup.hpp>
#include <ATen/ThreadLocalState.h>

#include <c10/util/Logging.h>

namespace c10d {

// std::thread* ProcessGroup::task_listener_;

void TaskListenLoop(ProcessGroup& group_) {
  while (TaskListenLoopOnce(group_)) {}
}

bool TaskListenLoopOnce(ProcessGroup& group_) {
  // printf("loop called\n");
  if (group_.dequeueTask()) {
    auto task = group_.getFrontTask();
    auto op = task->getOperation();
    auto opts = task->getOpts();
    auto data = task->getData();
    auto work = task->getWork();
    switch(op) {
      case OpType::BROADCAST:
      {
        printf("[listener GET] BROADCAST\n");
        work->real_work_ = group_.broadcast(*data, opts.broadcastOpts);
        // group_.fake_work_mapping[(void*)work.get()] = new_work;
      }
      case OpType::ALLREDUCE:
      {
        printf("[listener GET] ALLREDUCE workptr: %p\n", work.get());
        work->real_work_ = group_._allreduce(*data, opts.allreduceOpts);
        // group_.fake_work_mapping[(void*)work.get()] = new_work;
      }
      // case OpType::ALLREDUCE_COALESCED:
      //   return "ALLREDUCE_COALESCED";
      // case OpType::REDUCE:
      //   return "REDUCE";
      // case OpType::ALLGATHER:
      //   return "ALLGATHER";
      // case OpType::ALLGATHER_BASE:
      //   return "ALLGATHER_BASE";
      // case OpType::ALLGATHER_COALESCED:
      //   return "ALLGATHER_COALESCED";
      // case OpType::GATHER:
      //   return "GATHER";
      // case OpType::SCATTER:
      //   return "SCATTER";
      // case OpType::REDUCE_SCATTER:
      //   return "REDUCE_SCATTER";
      // case OpType::ALLTOALL_BASE:
      //   return "ALLTOALL_BASE";
      // case OpType::ALLTOALL:
      //   return "ALLTOALL";
      // case OpType::SEND:
      //   return "SEND";
      // case OpType::RECV:
      //   return "RECV";
      // case OpType::RECVANYSOURCE:
      //   return "RECVANYSOURCE";
      // case OpType::BARRIER:
      //   return "BARRIER";
      // case OpType::UNKNOWN:
      //   return "UNKNOWN";
      default:
        TORCH_INTERNAL_ASSERT("Unknown op type!");
    }
    // task->
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

std::string opTypeToString(OpType opType) {
  switch (opType) {
    case OpType::BROADCAST:
      return "BROADCAST";
    case OpType::ALLREDUCE:
      return "ALLREDUCE";
    case OpType::ALLREDUCE_COALESCED:
      return "ALLREDUCE_COALESCED";
    case OpType::REDUCE:
      return "REDUCE";
    case OpType::ALLGATHER:
      return "ALLGATHER";
    case OpType::ALLGATHER_BASE:
      return "ALLGATHER_BASE";
    case OpType::ALLGATHER_COALESCED:
      return "ALLGATHER_COALESCED";
    case OpType::GATHER:
      return "GATHER";
    case OpType::SCATTER:
      return "SCATTER";
    case OpType::REDUCE_SCATTER:
      return "REDUCE_SCATTER";
    case OpType::ALLTOALL_BASE:
      return "ALLTOALL_BASE";
    case OpType::ALLTOALL:
      return "ALLTOALL";
    case OpType::SEND:
      return "SEND";
    case OpType::RECV:
      return "RECV";
    case OpType::RECVANYSOURCE:
      return "RECVANYSOURCE";
    case OpType::BARRIER:
      return "BARRIER";
    case OpType::UNKNOWN:
      return "UNKNOWN";
    default:
      TORCH_INTERNAL_ASSERT("Unknown op type!");
  }
  return "UNKNOWN";
}

bool isP2POp(OpType opType) {
  return opType == OpType::SEND || opType == OpType::RECV ||
      opType == OpType::RECVANYSOURCE;
}


ProcessGroup::Work::Work(int rank, OpType opType, const char* profilingTitle)
    : rank_(rank), opType_(opType) {
  if (profilingTitle != nullptr) {
    auto recordingFunction = std::make_shared<at::RecordFunction>(at::RecordScope::USER_SCOPE);
    if (recordingFunction->isActive()) {
        recordingFunction->before(profilingTitle, {});
        std::function<void()> end_handler = [this, recordingFunction]() {
          recordingFunction->end();
        };
        recordFunctionEndCallback_ = at::wrapPropagateTLSState(end_handler);
    }
  }
}

OpType ProcessGroup::Work::retrieveOpType() {
  return opType_;
}

ProcessGroup::Work::~Work() {}

bool ProcessGroup::Work::isCompleted() {
  std::lock_guard<std::mutex> lock(mutex_);
  return completed_;
}

bool ProcessGroup::Work::isSuccess() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !exception_;
}

std::exception_ptr ProcessGroup::Work::exception() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return exception_;
}

int ProcessGroup::Work::sourceRank() const {
  throw std::runtime_error(
      "sourceRank() may only be called on work objects "
      "that correspond to a recv or recv-from-any call.");
}

std::vector<at::Tensor> ProcessGroup::Work::result() {
  throw std::runtime_error("result() not implemented.");
}

void ProcessGroup::Work::synchronize() {}

bool ProcessGroup::Work::wait(std::chrono::milliseconds timeout) {
  while (1) {
    if (this->real_work_) {
      // Lam: This work is already overwrited.
      this->real_work_->wait();
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return true;
}

bool ProcessGroup::Work::_wait(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (timeout == kNoTimeout) {
    // This waits without a timeout.
    cv_.wait(lock, [&] { return completed_; });
  } else {
    // Waits for the user-provided timeout.
    cv_.wait_for(lock, timeout, [&] { return completed_; });
    if (!completed_) {
      // Throw exception if the wait operation timed out and the work was not
      // completed.
      throw std::runtime_error("Operation timed out!");
    }
  }
  if (exception_) {
    std::rethrow_exception(exception_);
  }
  synchronize();
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroup::Work::abort() {
  TORCH_CHECK(false, "ProcessGroup::Work::abort not implemented.");
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroup::Work::getFuture() {
  TORCH_CHECK(false, "ProcessGroup::Work::getFuture not implemented.")
}

void ProcessGroup::Work::finish(std::exception_ptr exception) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_ = true;
  exception_ = exception;
  if (recordFunctionEndCallback_) {
    recordFunctionEndCallback_();
    recordFunctionEndCallback_ = nullptr;
  }
  lock.unlock();
  cv_.notify_all();
}

void ProcessGroup::Work::finishAndThrow(std::exception_ptr exception) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_ = true;
  exception_ = exception;
  if (recordFunctionEndCallback_) {
    recordFunctionEndCallback_();
    recordFunctionEndCallback_ = nullptr;
  }
  if (exception_) {
    std::rethrow_exception(exception_);
  }
}

ProcessGroup::CollectiveWork::CollectiveWork(
    c10d::CollectiveOptions opts,
    c10d::OpType operation,
    std::vector<at::Tensor>* data,
    c10::intrusive_ptr<c10d::ProcessGroup::Work> work,
    uint32_t priority)
    : opts_(opts),
      operation_(operation),
      data_(data),
      work_(work),
      priority_(priority){};

// c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroup::
//     wait_collective_queue(c10::intrusive_ptr<c10d::ProcessGroup::Work> work) {
//       while (1) {
//         auto iter = fake_work_mapping.find(work);
//         if (iter != fake_work_mapping.end())
//           return iter->second;
//         std::this_thread::sleep_for(std::chrono::milliseconds(10));
//       }
//     }

// c10::intrusive_ptr<c10d::ProcessGroup::Work> ProcessGroup::enqueueTask(
//     OpType operation,
//     std::vector<at::Tensor>& data,
//     c10::intrusive_ptr<c10d::ProcessGroup::Work> work) {
//       uint32_t priority = 0;
//       auto work_new = c10::make_intrusive<c10d::ProcessGroup::Work>();
//       // collective_queue_.emplace(c10::make_intrusive<ProcessGroup::CollectiveWork>(operation, &data, work_new, priority));
//       // Lam: This work is a fake work, will substitute later.
//       return work_new;
//     }

bool ProcessGroup::dequeueTask() {
  if (!collective_queue_.empty()) {
    front_task_ = collective_queue_.top();
    collective_queue_.pop();
    return true;
  } else {
    return false;
  }
}

ProcessGroup::ProcessGroup(int rank, int size)
    : rank_(rank), size_(size) {
  C10_LOG_API_USAGE_ONCE("c10d.process_group");
  task_listener_ = new std::thread(TaskListenLoop, std::ref(*this));
}

ProcessGroup::~ProcessGroup() {}

// This is introduced so that implementors of ProcessGroup would not need to
// have this implmentation.
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroup::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* usused */,
    std::vector<at::Tensor>& /* usused */,
    const AllgatherOptions& /* usused */) {
  throw std::runtime_error(
      "no support for allgather_coalesced in this process group");
}

} // namespace c10d
