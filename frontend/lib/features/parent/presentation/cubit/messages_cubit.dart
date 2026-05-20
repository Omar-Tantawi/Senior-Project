import 'package:equatable/equatable.dart';
import 'package:first_try/features/parent/data/models/parent_extra_models.dart';
import 'package:first_try/features/parent/data/repos/parent_repo.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

// ── States ───────────────────────────────────────────────────────────────────

sealed class MessagesState extends Equatable {
  const MessagesState();
  @override
  List<Object?> get props => [];
}

class MessagesInitial extends MessagesState {
  const MessagesInitial();
}

class MessagesLoading extends MessagesState {
  const MessagesLoading();
}

class MessagesLoaded extends MessagesState {
  final List<MessageModel> inbox;

  /// All teachers across every child — used when no child is selected.
  final List<TeacherSummaryModel> allTeachers;
  final bool allTeachersLoading;

  /// Teachers loaded for a specific child (keyed by childId).
  final Map<int, List<TeacherSummaryModel>> teachersByChild;

  /// Which child's teacher list is currently being fetched.
  final int? teachersLoadingForChild;

  final MessageModel? opened;
  final bool sending;

  const MessagesLoaded({
    required this.inbox,
    this.allTeachers = const [],
    this.allTeachersLoading = false,
    this.teachersByChild = const {},
    this.teachersLoadingForChild,
    this.opened,
    this.sending = false,
  });

  /// Teachers for a specific child, or empty if not yet loaded.
  List<TeacherSummaryModel> teachersFor(int childId) =>
      teachersByChild[childId] ?? [];

  bool isLoadingTeachersFor(int childId) =>
      teachersLoadingForChild == childId;

  MessagesLoaded copyWith({
    List<MessageModel>? inbox,
    List<TeacherSummaryModel>? allTeachers,
    bool? allTeachersLoading,
    Map<int, List<TeacherSummaryModel>>? teachersByChild,
    int? teachersLoadingForChild,
    bool clearTeachersLoading = false,
    MessageModel? opened,
    bool clearOpened = false,
    bool? sending,
  }) =>
      MessagesLoaded(
        inbox: inbox ?? this.inbox,
        allTeachers: allTeachers ?? this.allTeachers,
        allTeachersLoading: allTeachersLoading ?? this.allTeachersLoading,
        teachersByChild: teachersByChild ?? this.teachersByChild,
        teachersLoadingForChild: clearTeachersLoading
            ? null
            : (teachersLoadingForChild ?? this.teachersLoadingForChild),
        opened: clearOpened ? null : (opened ?? this.opened),
        sending: sending ?? this.sending,
      );

  @override
  List<Object?> get props => [
        inbox,
        allTeachers,
        allTeachersLoading,
        teachersByChild,
        teachersLoadingForChild,
        opened,
        sending,
      ];
}

class MessagesError extends MessagesState {
  final String message;
  const MessagesError(this.message);
  @override
  List<Object?> get props => [message];
}

// ── Cubit ────────────────────────────────────────────────────────────────────

class MessagesCubit extends Cubit<MessagesState> {
  final ParentRepo repo;

  MessagesCubit({required this.repo}) : super(const MessagesInitial());

  Future<void> loadInbox() async {
    emit(const MessagesLoading());
    try {
      final inbox = await repo.getMessages();
      emit(MessagesLoaded(inbox: inbox));
    } catch (e) {
      emit(MessagesError(e.toString()));
    }
  }

  /// Loads all teachers (no child filter). Used when no child is selected
  /// in the compose form. Skipped if already loaded or loading.
  Future<void> loadTeachers() async {
    final s = state;
    if (s is! MessagesLoaded) return;
    if (s.allTeachers.isNotEmpty || s.allTeachersLoading) return;

    emit(s.copyWith(allTeachersLoading: true));
    try {
      final teachers = await repo.getTeachers();
      final current = state;
      if (current is MessagesLoaded) {
        emit(current.copyWith(allTeachers: teachers, allTeachersLoading: false));
      }
    } catch (_) {
      final current = state;
      if (current is MessagesLoaded) {
        emit(current.copyWith(allTeachersLoading: false));
      }
    }
  }

  /// Loads teachers for a specific child (with their subjects).
  /// Results are cached by childId so repeated opens are instant.
  Future<void> loadTeachersForChild(int childId) async {
    final s = state;
    if (s is! MessagesLoaded) return;
    if (s.teachersByChild.containsKey(childId)) return;
    if (s.teachersLoadingForChild == childId) return;

    emit(s.copyWith(teachersLoadingForChild: childId));
    try {
      final teachers = await repo.getTeachersForChild(childId);
      final current = state;
      if (current is MessagesLoaded) {
        final updated =
            Map<int, List<TeacherSummaryModel>>.from(current.teachersByChild)
              ..[childId] = teachers;
        emit(current.copyWith(
          teachersByChild: updated,
          clearTeachersLoading: true,
        ));
      }
    } catch (_) {
      final current = state;
      if (current is MessagesLoaded) {
        emit(current.copyWith(clearTeachersLoading: true));
      }
    }
  }

  /// Open a specific message (fetches fresh so read_at is updated server-side).
  Future<void> open(int messageId) async {
    final s = state;
    final base = s is MessagesLoaded ? s : const MessagesLoaded(inbox: []);
    emit(base.copyWith(opened: null, clearOpened: true));
    try {
      final m = await repo.getMessage(messageId);
      final inbox = base.inbox.map((x) => x.id == m.id ? m : x).toList();
      emit(base.copyWith(inbox: inbox, opened: m));
    } catch (e) {
      emit(MessagesError(e.toString()));
    }
  }

  void closeOpened() {
    final s = state;
    if (s is MessagesLoaded) emit(s.copyWith(clearOpened: true));
  }

  Future<bool> send({
    required int teacherId,
    int? studentId,
    required String subject,
    required String body,
  }) async {
    final s = state;
    final base = s is MessagesLoaded ? s : const MessagesLoaded(inbox: []);
    emit(base.copyWith(sending: true));
    try {
      final created = await repo.sendMessage(
        teacherId: teacherId,
        studentId: studentId,
        subject: subject,
        body: body,
      );
      emit(base.copyWith(
        inbox: [created, ...base.inbox],
        sending: false,
      ));
      return true;
    } catch (e) {
      emit(base.copyWith(sending: false));
      emit(MessagesError(e.toString()));
      return false;
    }
  }
}
