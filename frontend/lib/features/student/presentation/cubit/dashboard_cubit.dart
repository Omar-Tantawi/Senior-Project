import 'package:first_try/features/student/data/mocks/student_mock_data.dart';
import 'package:first_try/features/student/data/models/student_models.dart';
import 'package:first_try/features/student/data/repos/student_repo.dart';
import 'package:first_try/features/student/presentation/cubit/dashboard_state.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class DashboardCubit extends Cubit<DashboardState> {
  final StudentRepo repo;

  DashboardCubit({required this.repo}) : super(DashboardInitial());

  Future<void> load() async {
    emit(DashboardLoading());

    // Get the student's name from profile first (required for the greeting).
    final StudentProfileModel profile;
    try {
      profile = await repo.getProfile();
    } catch (_) {
      emit(DashboardLoaded(StudentMockData.dashboard));
      return;
    }

    // Derive stats from individual endpoints in parallel; default to 0 on failure
    // so a new student with no data still sees their real name.
    int todayClasses = 0, unread = 0, hwCount = 0;
    double attendancePct = 0.0;

    try {
      final results = await Future.wait([
        repo.getAttendance(),
        repo.getSchedule(),
        repo.getNotifications(),
        repo.getHomework(),
      ]);

      final attendance    = results[0] as AttendanceSummaryModel;
      final schedule      = results[1] as List<ScheduleSlotModel>;
      final notifications = results[2] as List<NotificationModel>;
      final homework      = results[3] as List<HomeworkModel>;

      final todayKey = _todayKey();
      todayClasses   = schedule.where((s) => s.day == todayKey).length;
      unread         = notifications.where((n) => !n.isRead).length;
      hwCount        = homework
          .where((h) => h.status == 'pending' || h.status == 'late')
          .length;
      attendancePct  = attendance.percent;
    } catch (_) {
      // Stats unavailable — show real name with zeroed counts.
    }

    emit(DashboardLoaded(StudentDashboardModel(
      name: profile.name,
      todayClassesCount: todayClasses,
      unreadNotificationsCount: unread,
      attendancePercent: attendancePct,
      upcomingHomeworkCount: hwCount,
    )));
  }

  String _todayKey() {
    const keys = [
      'monday', 'tuesday', 'wednesday', 'thursday',
      'friday', 'saturday', 'sunday',
    ];
    return keys[(DateTime.now().weekday - 1).clamp(0, 6)];
  }
}
