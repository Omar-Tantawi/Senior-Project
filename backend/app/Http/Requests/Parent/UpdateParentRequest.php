<?php

namespace App\Http\Requests\Parent;

use App\Models\Guardian;
use Illuminate\Foundation\Http\FormRequest;

class UpdateParentRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        $userId = Guardian::where('parent_id', $this->route('id'))->firstOrFail()->user_id;

        return [
            'name'      => 'sometimes|string|max:255',
            'email'     => "sometimes|email|unique:users,email,{$userId}",
            'phone'     => 'sometimes|nullable|string|max:20',
            'is_active' => 'sometimes|boolean',
        ];
    }
}
