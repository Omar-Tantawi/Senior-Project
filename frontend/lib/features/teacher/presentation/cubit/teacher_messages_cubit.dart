import 'package:equatable/equatable.dart';
import 'package:first_try/features/teacher/data/models/teacher_extra_models.dart';
import 'package:first_try/features/teacher/data/repos/teacher_repo.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

// ── States ───────────────────────────────────────────────────────────────────

sealed class TeacherMessagesState extends Equatable {
  const TeacherMessagesState();
  @override
  List<Object?> get props => [];
}

class TeacherMessagesInitial extends TeacherMessagesState {
  const TeacherMessagesInitial();
}

class TeacherMessagesLoading extends TeacherMessagesState {
  const TeacherMessagesLoading();
}

class TeacherMessagesLoaded extends TeacherMessagesState {
  final int unreadCount;
  final List<TeacherMessageModel> inbox;
  final List<TeacherMessageModel> sent;
  final TeacherMessageModel? opened;
  final bool sending;

  /// Parents of the teacher's students — populated lazily when compose opens.
  final List<ParentSummaryModel> parents;
  final bool parentsLoading;

  const TeacherMessagesLoaded({
    this.unreadCount = 0,
    this.inbox = const [],
    this.sent = const [],
    this.opened,
    this.sending = false,
    this.parents = const [],
    this.parentsLoading = false,
  });

  TeacherMessagesLoaded copyWith({
    int? unreadCount,
    List<TeacherMessageModel>? inbox,
    List<TeacherMessageModel>? sent,
    TeacherMessageModel? opened,
    bool clearOpened = false,
    bool? sending,
    List<ParentSummaryModel>? parents,
    bool? parentsLoading,
  }) =>
      TeacherMessagesLoaded(
        unreadCount: unreadCount ?? this.unreadCount,
        inbox: inbox ?? this.inbox,
        sent: sent ?? this.sent,
        opened: clearOpened ? null : (opened ?? this.opened),
        sending: sending ?? this.sending,
        parents: parents ?? this.parents,
        parentsLoading: parentsLoading ?? this.parentsLoading,
      );

  @override
  List<Object?> get props =>
      [unreadCount, inbox, sent, opened, sending, parents, parentsLoading];
}

class TeacherMessagesError extends TeacherMessagesState {
  final String message;
  const TeacherMessagesError(this.message);
  @override
  List<Object?> get props => [message];
}

// ── Cubit ────────────────────────────────────────────────────────────────────

class TeacherMessagesCubit extends Cubit<TeacherMessagesState> {
  final TeacherRepo repo;

  TeacherMessagesCubit({required this.repo})
      : super(const TeacherMessagesInitial());

  Future<void> load() async {
    emit(const TeacherMessagesLoading());
    try {
      final results = await Future.wait([
        repo.getInbox(),
        repo.getSentMessages(),
      ]);
      final inbox = results[0] as TeacherInboxModel;
      final sent  = results[1] as List<TeacherMessageModel>;
      emit(TeacherMessagesLoaded(
        unreadCount: inbox.unreadCount,
        inbox: inbox.messages,
        sent: sent,
      ));
    } catch (e) {
      emit(TeacherMessagesError(e.toString()));
    }
  }

  /// Lazy-loads the parent list for the compose picker.
  /// Skipped if already loaded or loading.
  Future<void> loadParents() async {
    final s = state;
    if (s is! TeacherMessagesLoaded) return;
    if (s.parents.isNotEmpty || s.parentsLoading) return;

    emit(s.copyWith(parentsLoading: true));
    try {
      final parents = await repo.getParents();
      final current = state;
      if (current is TeacherMessagesLoaded) {
        emit(current.copyWith(parents: parents, parentsLoading: false));
      }
    } catch (_) {
      final current = state;
      if (current is TeacherMessagesLoaded) {
        emit(current.copyWith(parentsLoading: false));
      }
    }
  }

  Future<void> open(int messageId) async {
    final s = state;
    final base = s is TeacherMessagesLoaded ? s : const TeacherMessagesLoaded();
    emit(base.copyWith(clearOpened: true));
    try {
      final m = await repo.getMessage(messageId);
      // Reflect read status in inbox list.
      final inbox = base.inbox.map((x) => x.id == m.id ? m : x).toList();
      final unread = inbox.where((x) => !x.isRead).length;
      emit(base.copyWith(opened: m, inbox: inbox, unreadCount: unread));
    } catch (e) {
      emit(TeacherMessagesError(e.toString()));
    }
  }

  void closeOpened() {
    final s = state;
    if (s is TeacherMessagesLoaded) emit(s.copyWith(clearOpened: true));
  }

  Future<bool> send({
    required int receiverUserId,
    int? studentId,
    String? subject,
    required String body,
  }) async {
    final s = state;
    final base = s is TeacherMessagesLoaded ? s : const TeacherMessagesLoaded();
    emit(base.copyWith(sending: true));
    try {
      final created = await repo.sendMessage(
        receiverUserId: receiverUserId,
        studentId: studentId,
        subject: subject,
        body: body,
      );
      emit(base.copyWith(
        sent: [created, ...base.sent],
        sending: false,
      ));
      return true;
    } catch (e) {
      emit(base.copyWith(sending: false));
      emit(TeacherMessagesError(e.toString()));
      return false;
    }
  }
}
