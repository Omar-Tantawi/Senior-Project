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

    // Step 1: name comes from profile — if this fails we have nothing useful to show.
    final StudentProfileModel profile;
    try {
      profile = await repo.getProfile();
    } catch (_) {
      emit(DashboardLoaded(StudentMockData.dashboard));
      return;
    }

    // Step 2: stats — each call is independent so one failure cannot zero out the rest.
    int todayClasses = 0, unread = 0, hwCount = 0;
    double attendancePct = 0.0;

    await Future.wait([
      repo.getAttendance()
          .then((a) => attendancePct = a.percent)
          .catchError((_) {}),

      repo.getSchedule().then((slots) {
        final todayKey = _todayKey();
        todayClasses = slots.where((s) => s.day == todayKey).length;
      }).catchError((_) {}),

      repo.getNotifications().then((notifs) {
        unread = notifs.where((n) => !n.isRead).length;
      }).catchError((_) {}),

      repo.getHomework().then((hw) {
        hwCount = hw
            .where((h) => h.status == 'pending' || h.status == 'late')
            .length;
      }).catchError((_) {}),
    ]);

    emit(DashboardLoaded(StudentDashboardModel(
      name: profile.name,
      todayClassesCount: todayClasses,
      unreadNotificationsCount: unread,
      attendancePercent: attendancePct,
      upcomingHomeworkCount: hwCount,
    )));
  }

  /// Matches ScheduleCubit._todayKey(): school week is Sun–Thu, so Friday and
  /// Saturday snap forward to Sunday (the next school day).
  String _todayKey() {
    const keys = [
      'monday', 'tuesday', 'wednesday', 'thursday',
      'friday', 'saturday', 'sunday',
    ];
    const schoolDays = {
      'sunday', 'monday', 'tuesday', 'wednesday', 'thursday',
    };
    final today = keys[(DateTime.now().weekday - 1).clamp(0, 6)];
    return schoolDays.contains(today) ? today : 'sunday';
  }
}
