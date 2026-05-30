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
    try {
      // No dedicated dashboard endpoint yet — surface the real name from
      // `/student/{id}/profile` and fill the rest from mock until a proper
      // dashboard endpoint exists (today's classes count, pending homework,
      // attendance percent).
      final profile = await repo.getProfile();
      const mock = StudentMockData.dashboard;
      emit(DashboardLoaded(StudentDashboardModel(
        name: profile.name,
        todayClassesCount:        mock.todayClassesCount,
        unreadNotificationsCount: mock.unreadNotificationsCount,
        attendancePercent:        mock.attendancePercent,
        upcomingHomeworkCount:    mock.upcomingHomeworkCount,
      )));
    } catch (_) {
      // Profile fetch failed — fall back to fully-mock dashboard.
      emit(DashboardLoaded(StudentMockData.dashboard));
    }
  }
}
